# PointNetVlad-FiLM implementation
Implementation of PointNetVlad-FiLM, aka [PointNetVLAD](https://github.com/mikacuy/pointnetvlad) with the Conditional Batch Normalization layer from [FiLM-Ensemble](https://github.com/prs-eth/FILM-Ensemble).

## PAPER 
[PointNetVLAD](https://arxiv.org/abs/1804.03492)

[FiLM-Ensemble](https://arxiv.org/abs/2206.00050)

## INTRODUCTION
The entire codebase is an adaption from the Pytorch implementation of PointNetVLAD from [@cattaneod](https://github.com/cattaneod/PointNetVlad-Pytorch).

The main scope of this project is to implement an implicit ensemble model, using the CBN layer described in FiLM-Ensemble, for a task of uncertainty estimation in 2D or 3D place recognition for my master thesis.

I chose to use the PointNetVLAD architecture because it is a simple architecture and I could easily manipulate. 
I would like to try different architectures, but I'm not sure if I'll have the time to do so.

I'm still working on the code, so it's not yet ready for use. I'll update this README when it is.

## MAIN ISSUES
- I don't know why the script load 25GB of data in RAM, lowering the number of queries doesn't help. This is a huge BOH
- The RTX 1080 lack of memory when I try to train the ensemble (>1 element). I tried with distributed training, but it doesn't work, it didn't distribute the data on the GPUs. ANOTHER HUGE BOH

## MILESTONES
- [x] Implement FiLM-Ensemble
- [] make it work with an ensemble of more than 1 element!
- [] make a proper dataloader (jeez this is a mess, imho)
- [] optimize the code
- [] experiment with different architectures