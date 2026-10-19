"""Exact frozen radial-network directional derivative along the edge length."""
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _paired_silu(X, Y, TOTAL: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    value = tl.load(X+index, index<TOTAL, 0.)
    tangent = tl.load(X+TOTAL+index, index<TOTAL, 0.)
    sigmoid = 1./(1.+tl.exp(-value))
    tl.store(Y+index, value*sigmoid*SCALE, index<TOTAL)
    tl.store(Y+TOTAL+index, tangent*SCALE*sigmoid*(1.+value*(1.-sigmoid)), index<TOTAL)


class RadialJVP:
    def __init__(self, fc, *, r_max, number_of_basis, function_type, polynomial_cutoff_p,
                 sqrt_num_basis_norm):
        self.fc = fc
        self.options = dict(r_max=r_max, number_of_basis=number_of_basis,
            function_type=function_type, polynomial_cutoff_p=polynomial_cutoff_p,
            sqrt_num_basis_norm=sqrt_num_basis_norm)
        self.supported = (function_type.lower() in ("bessel", "gaussian") and len(fc) == 7
            and all(isinstance(fc[k], torch.nn.Linear) and fc[k].bias is None for k in (0, 2, 4, 6)))

    def eligible(self):
        return self.supported and not self.fc.training and not any(p.requires_grad for p in self.fc.parameters())

    @property
    def weight(self):
        return self.fc[-1].weight

    def reference(self, distance):
        from chorus.models.radial_basis import mace_radial_embedding
        return self.fc(mace_radial_embedding(distance, **self.options))

    def evaluate(self, distance):
        from chorus.models.radial_basis import mace_radial_embedding, mace_polynomial_cutoff
        if not self.eligible():
            raise ValueError("Radial JVP requires the original frozen bias-free MLP")
        options = self.options
        radius, basis = options["r_max"], options["number_of_basis"]
        r = distance[:, None]
        if options["function_type"].lower() == "bessel":
            frequency = math.pi/radius*torch.linspace(1., float(basis), basis,
                device=distance.device, dtype=distance.dtype)
            denominator = r.clamp_min(1e-12)
            sine, cosine = torch.sin(r*frequency), torch.cos(r*frequency)
            raw = math.sqrt(2./radius)*sine/denominator
            derivative = math.sqrt(2./radius)*(cosine*frequency/denominator
                -sine/denominator.square()*(r>=1e-12))
        else:
            centre = torch.linspace(0., radius, basis, device=distance.device, dtype=distance.dtype)
            coefficient = -0.5/(radius/float(max(basis-1, 1)))**2
            raw = torch.exp(coefficient*(r-centre).square())
            derivative = raw*(2.*coefficient*(r-centre))
        cutoff = options["polynomial_cutoff_p"]
        if cutoff is not None:
            p = int(cutoff)
            t = distance/radius
            envelope = mace_polynomial_cutoff(distance, radius, p)
            delta = (-((p+1.)*(p+2.)/2.)*p*t.pow(p-1)
                +p*(p+2.)*(p+1.)*t.pow(p)
                -(p*(p+1.)/2.)*(p+2.)*t.pow(p+1))/radius
            delta = delta*(distance<radius)
            derivative = derivative*envelope[:, None]+raw*delta[:, None]
        if options["sqrt_num_basis_norm"]:
            derivative = derivative*math.sqrt(basis)
        # Keep the model's literal embedding for the value, not an alternate
        # factorization of the sine/envelope product.
        value = mace_radial_embedding(distance, **options)
        paired = torch.cat((value, derivative), dim=0)
        for index in (0, 2, 4):
            linear = F.linear(paired, self.fc[index].weight)
            paired = torch.empty_like(linear)
            total = linear.numel()//2
            if total:
                _paired_silu[(triton.cdiv(total, 256),)](linear, paired, total,
                    self.fc[index+1]._scale, 256, enable_fp_fusion=False)
        return paired[:distance.numel()], paired[distance.numel():]
