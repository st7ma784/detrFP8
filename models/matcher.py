# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
import matplotlib.pyplot as plt

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        self.counter=0
        self.m=torch.distributions.normal.Normal(torch.tensor(-5.5),torch.tensor(2.5))
        self.tgt=torch.distributions.half_normal.HalfNormal(0.3)

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1) #Dist=N(6,3)  Ideal is 1-|N(0,0.2)|
        self.counter+=1
        # if self.counter%1000==0:
        #     plt.hist(C.flatten().numpy(),bins=100,label=self.counter)
        #     plt.title("Histogram at {}".format(self.counter))
        #     plt.savefig("Histogram at {}.png".format(self.counter))
        
        #optimize C for FP8
        #print("C Shape is {}".format(C.shape))   B,100,Annotations
        #print("sizes is {}".format(sizes)) # Sizes is B numbers that sum to A 

        #print("output is {}".format(indices)) (B * ([0-100]*s, range(0,s)))
        sizes = [len(v["boxes"]) for v in targets]
        #indices = [linear_sum_assignment(c[i],maximize=False) for i, c in enumerate(C.cpu().split(sizes, -1))]#C.shape is... 


        '''
        Tests,
        Just use argmin as is,  #3:39/epoch
        '''
        # E=torch.argmin(C,dim=1) #B,A 
        # indexes=[torch.arange(s) for s in sizes]
        # #print(C.split(sizes,-1))
        # locations=E[torch.diag(torch.ones(E.shape[0],dtype=torch.bool)).repeat_interleave(torch.tensor(sizes,dtype=torch.long),dim=1)].split(sizes)
        # indices=list(zip(locations,indexes))
        # print(indices)


        '''
        Use argmax after FP8...
        
        
        '''

        # prob=self.m.cdf(-C) # using - so that the minimize LSA can become a maximize function!
        # reDistC=self.tgt.icdf(prob)

        # C=torch.argmax(reDistC,dim=1) #B,A 
        # indexes=[torch.arange(s) for s in sizes]
        # #print(C.split(sizes,-1))
        # locations=C[torch.diag(torch.ones(C.shape[0],dtype=torch.bool)).repeat_interleave(torch.tensor(sizes,dtype=torch.long),dim=1)].split(sizes)
        # indices=list(zip(locations,indexes))
        # # print(indices)

        '''
        
        use stepped argmax as per my_function
        '''
        S=torch.tensor(sizes,dtype=torch.long,device=C.device)
        prob=self.m.cdf(-C) # using - so that the minimize LSA can become a maximize function!
        fx=self.tgt.icdf(prob).permute(0,2,1)[torch.diag(torch.ones_like(S,dtype=torch.bool,device=C.device)).repeat_interleave(S,dim=1)]
        row_lookups=lookup_sizes_vfast(S)
        outputs=Batch_MyLinearSumAssignment(fx.T,row_lookups=row_lookups)
        indices=[torch.nonzero(o,as_tuple=True) for o in outputs.split(sizes,dim=1)]
        return indices#[(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)

def Batch_MyLinearSumAssignment(Batched_TruthTensor:torch.Tensor,row_lookups:torch.Tensor, maximize=True,lookahead=2):
    # results=torch.zeros_like(Batched_TruthTensor,dtype=torch.bool)    
    for _ in range(row_lookups.shape[-1]): # number of columns 
        deltas=torch.diff(torch.topk(torch.clamp(Batched_TruthTensor,max=100),lookahead,dim=0,largest=maximize).values,n=lookahead-1,dim=0)
        col_index=torch.argmax(torch.abs(deltas[0])) 
        row_index=torch.argmax(Batched_TruthTensor[:,col_index],dim=0) # BxB
        Batched_TruthTensor[:,col_index]=0
        Batched_TruthTensor[row_index,row_lookups[col_index]]=0
        Batched_TruthTensor[row_index,col_index]=-1e-7
    return Batched_TruthTensor<0


@torch.no_grad()
def lookup_sizes_vfast(sizes:torch.Tensor):
    return torch.diag(torch.ones_like(sizes)).repeat_interleave(sizes,dim=0).repeat_interleave(sizes,dim=1).to(dtype=torch.long,device=sizes.device)



