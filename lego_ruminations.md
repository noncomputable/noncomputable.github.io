### Ruminations on Lego Intelligence

Why go through all that, when we could just train a small neural network on task specific data? Furthermore, if we got a neural rationale from a big, general model, couldn't we just retrain it on task specific data to get better performance? And wouldn't doing that break its circuits, undermining our goal of "interpretable building blocks"?

We are comparing 3 options here:

_A_. Train a small neural network on task specific data.<br/>
_B_. Extract a rationale from a big neural network as is.<br/>
_C_. Extract a rationale from a big neural network and retrain it on task specific data.<br/>

The weak case for lego intelligence is that B is better than A. The strong case is that B is better than A _and_ C.
When I say X is "better" than Y, I mean that I expect an attempt to do X will result in a model with _at least as good_ performance as an attempt to do Y **and** X is more interpretable. In every case, B allows us to use building blocks from a standard catalogue of well understood neural rationales, which is a step better for interpretability than either A or C can allow.

#### Thoughts on the Weak Case

Is B better than A? For A, task specific data has to be curated and training has to be carefully controlled and optimized to achieve good generalization. On the other hand, if there exists a large model that has been trained on a massive dataset and contains a rationale that captures the task in question, it's likely to generalize better than a small neural network trained on curated, task specific data. For example, without hand-crafted inductive biases, a small model trained on photos of apples might fail to identify a sketch of an apple, while a large, general model trained on both apples and sketches and thousands of other classes could succeed.

#### Thoughts on the Strong Case

Is B better than C? For the same reason as the weak case, task specific data could bias the network away from the generally useful features and circuits it learned from being trained on a massive, general dataset. So if anything, I would expect retraining it on a smaller, curated dataset to often negatively affect generalization to the true task.

#### The Big Counterargument

How are we identifying rationales inside the big neural network in the first place? We're doing a tree search whose objective is the rationale's relationship with a task. How do we measure that relationship? By comparing the output of the circuit with task specific data. What task specific data? Yes, that same task specific data that I said was going to irredeemably bias the rationale in A and C. So just like retraining the rationale on task specific data would bias it post-hoc, why wouldn't steering our search with task specific data bias the rationale ad-hoc just the same? If our algorithm is truly finding the _minimal substructure_ that has a high property score for this task specific data, then wouldn't it exclude the more generally useful circuits in the big network, and include only the circuits that overfit our data? This feels like a big issue.

For now, I will console myself with 2 ideas:

1. The features that capture our high level task specific data will be deeper in the network, and the features that are useful for generalizing to the true task will be earlier in the network, and so the circuits that we extract will tend to contain transformations that are useful for generalizing to the true task, even if we are steering search with a biased objective.
2. Even if B results in a rationale as biased ad-hoc as A and C are biased post-hoc, I don't see any reason why it would be _more_ biased. So if its performance is the same, B still has one, massive upside: it's made out of interpretable building blocks that we can catalogue and standardize.

#### Conclusions

It's very possible that all my intuitions are wrong. Maybe there's some reason this approach will be impossible or terrible in practice, I would love to find out. Otherwise, the only way to move forward and see is to run experiments on A and C VS B.
