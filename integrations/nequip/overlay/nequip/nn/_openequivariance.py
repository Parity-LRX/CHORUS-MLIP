"""Optional OpenEquivariance tensor products for the NequIP interaction.

CHORUS consumes per-edge equivariant messages before neighbor reduction.  The
standalone OpenEquivariance tensor product therefore provides the appropriate
acceleration boundary: it replaces the e3nn tensor-product kernel while
preserving the edge-message tensor needed by both Final and Persistent CHORUS.
"""

from __future__ import annotations

import torch
from e3nn import o3


class OpenEquivarianceTensorProduct(torch.nn.Module):
    """Parameter-free OpenEquivariance replacement for an e3nn TensorProduct."""

    def __init__(
        self,
        feature_irreps_in,
        irreps_edge_attr,
        irreps_mid,
        instructions,
        fused_convolution: bool = False,
    ) -> None:
        super().__init__()
        try:
            from openequivariance import (
                TPProblem,
                TensorProduct,
                TensorProductConv,
                torch_to_oeq_dtype,
            )
        except ImportError as exc:
            raise ImportError(
                "openequivariance_enabled=True requires the "
                "'openequivariance' package"
            ) from exc

        model_dtype = torch.get_default_dtype()
        problem = TPProblem(
            feature_irreps_in,
            irreps_edge_attr,
            irreps_mid,
            instructions,
            irrep_dtype=torch_to_oeq_dtype(model_dtype),
            weight_dtype=torch_to_oeq_dtype(model_dtype),
            shared_weights=False,
            internal_weights=False,
        )
        self.tensor_product = TensorProduct(
            problem,
            torch_op=True,
            use_opaque=False,
        )
        self.model_dtype = model_dtype
        self.use_fused_convolution = bool(fused_convolution)
        self.convolution = None
        if self.use_fused_convolution:
            self.convolution = TensorProductConv(
                problem, torch_op=True, deterministic=False, use_opaque=False
            )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_attributes: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Explicit casts make the strict-float32 contract robust to ambient AMP.
        return self.tensor_product(
            node_features.to(self.model_dtype),
            edge_attributes.to(self.model_dtype),
            edge_weights.to(self.model_dtype),
        )

    def aggregate(
        self,
        node_features: torch.Tensor,
        edge_attributes: torch.Tensor,
        edge_weights: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_src: torch.Tensor,
    ) -> torch.Tensor:
        """Native gather/TP/scatter when no edgewise CHORUS branch is needed."""
        if self.convolution is None:
            raise RuntimeError('Fused OEQ convolution was not configured')
        return self.convolution(
            node_features.to(self.model_dtype),edge_attributes.to(self.model_dtype),
            edge_weights.to(self.model_dtype),edge_dst,edge_src
        )


class OpenEquivarianceHermitianTensorProduct(torch.nn.Module):
    """Equal-rank Hermitian paths with rank as a batch dimension.

    Each orbital rank is independent in the original channelwise quadratic
    form. A singleton-multiplicity OEQ problem therefore needs only scalar
    +/-1 weights, rather than a dense R x R identity bank. No rank, CG path,
    parameter or relative-phase term is removed.
    """

    def __init__(self, irreps_charged, paths) -> None:
        super().__init__()
        from openequivariance import TPProblem,TensorProduct,torch_to_oeq_dtype

        charged=o3.Irreps(irreps_charged)
        if not charged or len({int(ir.l) for _,ir in charged})!=len(charged):
            raise ValueError('OEQ Hermitian requires one block per angular degree')
        ranks={int(mul) for mul,_ in charged}
        if len(ranks)!=1:
            raise ValueError('OEQ Hermitian requires a shared channelwise rank')
        self.rank=next(iter(ranks))
        self.lmax=max(int(ir.l) for _,ir in charged)
        if [int(ir.l) for _,ir in charged]!=list(range(self.lmax+1)):
            raise ValueError('OEQ Hermitian requires contiguous ordered angular degrees')
        self.input_slices=tuple(charged.slices())
        self.input_sizes=tuple(int(ir.dim) for _,ir in charged)
        singleton=o3.Irreps([(1,ir) for _,ir in charged])
        input_irreps=singleton+singleton
        nblocks=len(charged)
        output_irreps=[];instructions=[];weights=[];sizes=[]
        # Group directly in the same out-l/path/rank/m order as density_blocks.
        for out_l in range(self.lmax+1):
            count=0
            for l1,l2,target,component,_ in paths:
                if target!=out_l:continue
                slot=len(output_irreps)
                output_irreps.append((1,(out_l,1 if out_l%2==0 else -1)))
                if component=='real':
                    terms=((l1,l2,1.0),(nblocks+l1,nblocks+l2,1.0))
                elif component=='imag':
                    terms=((nblocks+l1,l2,1.0),(l1,nblocks+l2,-1.0))
                else:
                    raise ValueError('Unknown Hermitian path component')
                for left,right,weight in terms:
                    instructions.append((left,right,slot,'uvu',True))
                    weights.append(weight)
                count+=1
            sizes.append(count*(2*out_l+1))
        self.output_sizes=tuple(sizes)
        problem=TPProblem(input_irreps,input_irreps,o3.Irreps(output_irreps),instructions,
            irrep_dtype=torch_to_oeq_dtype(torch.get_default_dtype()),
            weight_dtype=torch_to_oeq_dtype(torch.get_default_dtype()),
            shared_weights=True,internal_weights=False,
            irrep_normalization='component',path_normalization='none')
        if problem.weight_numel!=len(weights):
            raise RuntimeError('Singleton OEQ Hermitian weight layout mismatch')
        self.tensor_product=TensorProduct(problem,torch_op=True,use_opaque=False)
        self.register_buffer('constant_weights',torch.tensor(weights),persistent=False)

    def forward(self,charged_real:torch.Tensor,charged_imag:torch.Tensor):
        count=charged_real.shape[0]
        pieces=[]
        for value in (charged_real,charged_imag):
            sections=torch.split(value,[self.rank*size for size in self.input_sizes],dim=-1)
            for section,size in zip(sections,self.input_sizes):
                pieces.append(section.reshape(count,self.rank,size))
        packed=torch.cat(pieces,dim=-1).reshape(count*self.rank,2*sum(self.input_sizes))
        coupled=self.tensor_product(packed,packed,self.constant_weights.to(
            dtype=packed.dtype,device=packed.device))
        outputs={};offset=0
        for out_l,size in enumerate(self.output_sizes):
            dim=2*out_l+1
            block=coupled[:,offset:offset+size].reshape(count,self.rank,size//dim,dim)
            outputs[out_l]=block.transpose(1,2).reshape(count,(size//dim)*self.rank,dim)
            offset+=size
        return outputs
