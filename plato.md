﻿---
title: Plato's Puppeteer - Measuring How Design Affects Decisionmaking
---
{:center: style="text-align: center"}
# Plato’s Puppeteer
{:center}
{:center: style="text-align: center"}
### Measuring How Design Affects Decisionmaking
{:center}
---
{:center: style="text-align: center"}
1\. [Introduction](#introduction)  
2\. [Eliciting User Beliefs](#eliciting-user-beliefs)  
3\. [Information Adversity](#information-adversity)  
4\. [Why](#why)
{:center}
---
#### **Introduction**

In 2016, after protests against its role in spreading disinformation, Facebook [announced](https://newsroom.fb.com/news/2016/12/news-feed-fyi-addressing-hoaxes-and-fake-news/) "disputed flags": links voted untrustworthy by fact checkers would be stamped with little red flags, effectively marking them "fake news". Then they [shut it down](https://techcrunch.com/2017/12/20/facebook-will-ditch-disputed-flags-on-fake-news-and-display-links-to-trustworthy-articles-instead/). They [reported](https://medium.com/facebook-design/designing-against-misinformation-e5846b3aa1e2) that the flags sometimes backfired and strengthened people's beliefs in the misinformation.

How didn't they figure this out before implementing it? Why not much sooner than nearly a year after the fact? Their final report pointed back to a [paper from 2012](https://journals.sagepub.com/doi/abs/10.1177/1529100612451018?journalCode=psia) on the psychology of misinformation for suggestions as to why it didn't work out.

How can they improve? Reading up on prior research _before_ proliferating big interface changes is a start. Innocuous features, like reversing the order of text, might have big, [documented effects](https://onlinelibrary.wiley.com/doi/full/10.1002/jcpy.1053). But that's also shaky. The 2012 paper Facebook cited in their retrospective referenced an earlier, general result on the backfire effect; in 2018, a [larger study](https://link.springer.com/article/10.1007/s11109-018-9443-y) failed to replicate that result among others. Due at least partly to [perverse incentives in academic publishing](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0020124), that a social or behavioral result is documented in a journal [doesn't appear to be a very strong signal that it's true](https://www.nature.com/articles/s41562-018-0399-z). 

But this isn't fatalistic. If researchers have a serious stake in the accuracy of their results, like those at big, influential firms should, they can design more reliable studies [for the specific features they're proposing](https://link.springer.com/article/10.1007/s11109-019-09533-0) and report more accurate results. Results that suggest how interface features might affect users' wellbeing.

Optimizing for a handful of "wellbeing" metrics can distract from less easily measurable objectives like [users' values](https://medium.com/what-to-build/is-anything-worth-maximizing-d11e648eb56f), but there are some measures of wellbeing that I think should influence (though not _determine_) how features get proliferated. Candidates for these have been studied in [online](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.22.7276&rep=rep1&type=pdf) [marketplaces](https://www.ifi.uzh.ch/ce/publications/BehavioralFactorsInMarketUIDesign.pdf) and [social media platforms](https://www.pnas.org/content/pnas/111/24/8788.full.pdf), looking at their effects on decision-making, [empathy & vitriol](https://qz.com/1264547/facebooks-problems-can-be-solved-with-design/), and [belief formation](https://onlinelibrary.wiley.com/doi/abs/10.1111/jcom.12166).

I'm interested in how interacting with a system affects people's ability to make good, informed decisions. How far is an interface from letting them make the best informed decisions possible? In other words, what's the _information adversity_ of the interface? The higher the interface's information adversity, the worse their decisions will be. Can we accurately, reliably measure it? Can we tell whether one feature is more or less _information adverse_ than another? I've sketched some basic tools for approaching these questions.

#### **Eliciting User Beliefs**

If we have an interface and some users, how can we measure its information adversity?

To measures information adversity, we have to assume that users' actions are based on their beliefs, with room for violations (e.g. accidental hand movements). What fits this pretty well? Consumption! Whether it's a physical product, an entertainment service, or transportation, people's consumption decisions are largely based on their beliefs.

We can run an experiment that goes like this: Enter information about different kinds of products into the system, then randomly assign each user an interface through which they can access that information. After they interact with it for some uniform time, give them a series of menus, let them nominate products of each kind, and give them a reward tied to the quality of each nomination—so the better the products they nominate, the higher their reward. Having recorded their decisions, and knowing the actual quality of the products in advance, we can calculate and compare the information adversity of different interfaces.

The details of each step can vary a lot with the application. Which products are used, how long users are allowed to interact, the structure and design of the menus, all depend on the context; the experimental design will have to be tailored for the specific users and interfaces in question. But what's the basic scaffolding we need to design these experiments?

First, we need products to enter into the system. In particular, we need sets of products $$ S$$ of the same category, or _substitute sets_, where we can be reasonably confident about whether any one is of higher or lower quality than any other in its set. Beyond that, the products can be any goods or services. We can come up with and give them artificial brand attributes and quality rankings.

The connection between interface features and user preferences might be confounded by their personal taste in brand attributes (e.g. the colors, graphics, and names associated with products), so we'd  randomly vary brand assignments between users.

We enter messages carrying information about the products into the system we're studying and let it expose them according to the rules of its interface. That can take the form of Facebook posts, Twitter bios, eBay listings, Amazon reviews, Google Sheets rows, Minecraft signposts, and so on. To control for the effects of different distributions of accuracy, we ought to generate messages so that their accuracy varies randomly across users & treatments.

Messages need to have certain properties to have a chance of being seen by the users we're studying, like being authored by someone they're connected to or being placed in a game region they frequent. We should randomly distribute these properties across messages to control for their exposure effects and isolate the effects of the interface features we're studying.

In the end, we want to know which products they _think_ are the best—limited by what the interface has shown them—and compare them to the ones that are actually the best. Our premise was that there are times when people's actions are driven by their beliefs, and we could name one of them: consumption. How can we get the users to act as consumers?

We let them nominate what they think are the best products in each substitute set. Then we tell them that those decisions won’t only give us information, but will provide them with products too. In summary: we give them one buying token for each substitute set, tell them to nominate their top products in $$ S$$, and that the token will be randomly spent on one of their nominations.

If they understand the product will be picked randomly from their nominations, the prospective quality of their decision (formally, its [subjective expected utility](https://en.wikipedia.org/wiki/Subjective_expected_utility)) will be the average of the prospective qualities (the qualities they believe they have, as opposed to actual qualities) of all their individual nominations. So we'd expect them to only nominate the products they think have the highest quality—though [in some settings people don't behave like this](https://en.wikipedia.org/wiki/Ellsberg_paradox), so we can't take this assumption for granted. Nominating products with lower prospective qualities than the maximum available will just lower the total prospective value of their decision, and so they'll only have an incentive to nominate the ones with the highest prospective quality in $$ S$$, i.e. their top products.

But that they'll tend to nominate only their top products doesn't mean they'll nominate _all_ of them. They may be averse to not knowing which specific product they'll get, even if they think the prospective qualities of all their top products are maximal. So they might just nominate one or a smaller, non-representative subset of their top products to escape this ambiguity. If the quality of only the one is far from the average quality of _all_ their top products, the measurement could seriously mislead us about their information adversity. So it's more important to make sure they follow the instructions, to nominate all their top products, as much as they can.

Like promised, we ought to give users a real product corresponding to a random one of their nominations for each set. Products should be [white-labeled](https://en.wikipedia.org/wiki/White-label_product) with the brand attributes they were artificially assigned. Choosing cheaper kinds of products will cheapen the cost of the experiment but will come at the cost of their incentive to act and think like consumers. We might also tweak the design of the experiment to cheapen it: instead of assigning one buying token to each substitute set, a smaller number of buying tokens are randomly distributed across a smaller number of substitute sets.

Giving them actual products also opens this study up to possible extensions. By using and experiencing what they thought would be their top products, they can update their beliefs. Through the interface, they might input new information about the products and see messages from other users, changing their decisions in subsequent rounds, experimentally informing us about its reputation mechanisms and their effect on information adversity.

If you don't have time for the definitions or details, but do have the data, I put together a simple tool to calculate information adversity [here](https://github.com/noncomputable/Information-Adversity).

#### **Information Adversity**

If we rank users' decisions by the quality of the products they pick, we can define the _information adversity_ of the interface as the difference between the ranks of the products they actually choose and the ranks of the best products available.

We'll define a ranking function $$ r(p)$$ for any product $$ p$$ to be the number of products that $$ p$$ has a higher quality than. If $$ r(A) > r(B)$$, $$ A$$ is preferred in quality to more products than $$ B$$. Since we're dealing only with the kinds of products that have reasonably solid quality distinctions (pens, chargers, tire replacements) and without ties, each product will have a unique rank.

If we want to make an accurate judgement about a person's behavior, we'll want to see their decisions across many different substitute sets $$ S$$. Where $$ r_S(p)$$ is the rank of a product $$ p$$ in $$ S$$, the minimum rank will be $$ 0$$ when $$ p$$ isn't preferred to any of the others, and the maximum rank will be $$ \vert S\vert - 1$$ when $$ p$$ is preferable to every product in $$ S$$ other than itself.

Whatever the actual rankings are, people's _beliefs_ about them are different. They might expect a product to have different qualities with different probabilities. In an experiment of the kind we discussed, we supposed that subjects will nominate products in $$ S$$ that have the highest prospective qualities, i.e. the ones they _think_ are the best.

Just like we chose to encode the products’ actual qualities into a ranking function, we can encode their prospective qualities into one too. For any product $$ p$$ in $$ S$$, $$ r^*_S(p)$$ will be the number of products that $$ p$$ has a higher prospective quality than. The subjective ranking function $$ r^*$$ will work the same way as $$ r$$, except it encodes information about beliefs instead of actual qualities.

However, there won't just be one universal subjective ranking function $$ r^*_S$$ for everyone with respect to $$ S$$. While we can be reasonably confident that people share the same rankings (i.e. each product pretty crisply has a higher or lower quality than each other), their subjective beliefs about those products can vary widely. Different people will think different products have different qualities with different probabilities. So different people (call them $$ i$$) will have different subjective ranking functions $$ r^*_{i,S}$$. $$ r^*_{i,S}(p)$$ will be the number of products in $$ S$$ that $$ p$$ has a higher prospective quality than _for person_ $$ i$$. Different people can have different beliefs, so subjective rankings will vary.

We want a measure of how far the products people choose are from the actual best products. We'll define an individual measure for each user and an index for the whole group.

We know the rank of the best product in $$ S$$ because we artificially specified it for the experiment, but what's the rank of the products a user _thinks_ are the best? They can have multiple top subjectively ranked products (multiple nominations in $$ S$$), so how can we give _one_ rank to all of them that we can compare to the maximum possible rank? Since as far as we know the user might consume any one of the nominated products with equal probability, our expectation about the rank of whatever they end up with will be some probabilistic mix of the actual ranks of all of them.

In particular, let $$ R^*_i(S)$$ be average of the actual ranks of the products in $$ S$$ that have the maximum subjective rank for person $$ i$$: $$ R^*_i(S) = \frac{\sum_{p \in M} r_S(p)}{\vert M\vert }$$, for $$ M = \underset{p}{\arg\max}\; r^*_{i,S}(p)$$. In other words, $$ R^*_i(S)$$ is the expected rank of their decision in $$ S$$. Since we suppose subjects will only nominate their top products in $$ S$$ (i.e. the ones with maximum subjective rank), in an experiment, $$ R^*_i(S)$$ will be the average of the actual ranks of the products they nominated.

The information adversity $$ I_i(S)$$ for one user $$ i$$ in one specific substitute set $$ S$$ might literally be how far the expected rank of their decision is from the rank of the best possible decision. That is, we could define it as the difference between the maximum product rank in $$ S$$, $$ R(S)$$, and the expected rank of $$ i$$'s decision in $$ S$$, $$ R^*_i(S)$$: $$ I_i(S) = R(S) - R^*_i(S)$$.

When the maximum rank $$ R(S)$$ and the user's expected rank $$ R^*_i(S)$$ are close, the information adversity $$ I_i(S)$$ will be low and closer to $$ 0$$; when they're far apart, $$ I_i(S)$$ will be higher and closer to $$ R(S)$$. Since $$ R(S)$$ is an upper bound on what $$ R^*_i(S)$$ can be, $$ I_i(S)$$ will always be a number between $$ 0$$ and $$ \vert S\vert -1$$; $$ I_i(S)$$ will only be meaningful in relation to $$ \vert S\vert $$. But we want to make a judgement about people's information adversity across multiple sets of products $$ S$$ of varying sizes $$ \vert S\vert$$.

We can get a measure of how far the expected rank is from the maximum rank that always falls between $$ 0$$ and $$ 1$$, independent of the size of $$ S$$, by finding the fraction their decision's actual expected rank $$ R^*_i(S)$$ is of the maximum possible rank $$ R(S)$$: $$ I_i(S) = \frac{R^*_i(S)}{R(S)}$$.

Since, again, $$ R(S)$$ is an upper bound on what $$ R^*_i(S)$$ can be, and both values will always be positive, $$ I_i(S)$$ will always be a value in $$ [0,1]$$. But when $$ R^*_i(S)$$ is low and far from $$ R(S)$$, the whole measure $$ I_i(S)$$ will be low. And when it's high and close to $$ R(S)$$, $$ I_i(S)$$ will be high. But that's the opposite of what we want: when the expected rank is far from the maximum possible rank, the information adversity should be high, and when they're close together, it should be low. Since it's in $$ [0,1]$$, to invert this pattern we can just redefine information adversity to be the complement of the previous definition: $$ I_i(S) = 1 - \frac{R^*_i(S)}{R(S)}$$.

That's person $$ i$$'s information adversity for one specific substitute set. How do we get an aggregate measure for *all* the $$ S$$ where $$ i$$ made nominations? We can find the fraction the sum of all expected ranks $$ R^*_i(S)$$ is of the sum of all maximum possible ranks $$ R(S)$$ for all sets $$ S$$: $$ I_i = 1 - \frac{\sum_{S} R^*_i(S)}{\sum_{S} R(S)}$$.

To keep things clear, let $$ R$$ be the sum of the maximum ranks of all substitute sets $$ S$$. Then let $$ R^*_i$$ be the sum of the expected ranks for all $$ S$$ for person $$ i$$: $$ R^*_i = \sum_{S} R^*_i(S)$$.

Then the aggregate measure of information adversity can be the average of the individual measures of all $$ n$$ people:

$$ I = 1 - \frac{1}{n}\sum_{i=1}^{n} \frac{R^*_i}{R} = \frac{\sum_{i=1}^{n} I_i}{n}$$

Information adversity is low when the ranks of the products people believe are the best are close to the ranks of the products that are actually the best. The further apart these values are, the higher the information adversity is.

If they do base their consumption decisions on their beliefs, they'll tend to choose products with the highest subjective ranks. Since rankings are determined by utilities and so differences in rank correspond to differences in utility, the information adversity will be a measure of welfare loss. It tries to answer the question of how far their decisions are from the most favorable ones possible. Given lots of choice data for lots of people, it puts a number on it, so we can compare levels of information adversity across different interfaces.

#### **Why**

Social media platforms like Twitter & Facebook and online marketplaces like eBay & Amazon aren't the end of the block. Far from it. I started out with the idea of a global [VRMMO](https://en.wikipedia.org/wiki/Virtual_reality) and discussion leaked into Minecraft signposts and office receptionists. As 3D virtual and augmented reality technologies become more popular, the number of wacky, unfamiliar interfaces that people can feasibly adopt will balloon.

As people rely more and more on these technologies to find and navigate information, consumption will become more and more embedded in them. They may be designed for any number of purposes with little concern for the welfare of their users and the markets embedded in them.

In Plato's _Allegory of the Cave_, prisoners are chained down in a cave, seeing its wall and nothing else their entire lives. A fire glows behind them, and in front of the fire sits a barrier behind which puppeteers walk unseen. They raise puppets above the barrier, casting shadows that the prisoners watch, discuss, and catalogue their entire lives. Plato posits that while the prisoners may know all about the shadows, they have no idea of the _true_ reality that lies behind them.

In a reality where important systems contain too much information to communicate in whole or at once, we're all constrained to get only a limited slice presented to us, just as the prisoners of the cave can only see shadows of the reality behind them. But how that slice is chosen and how it's presented to us matters enormously, just as what the puppeteers decide to raise before the flame matters enormously to what the prisoners believe. The people and organizations who craft and proliferate interface designs, just like the puppeteers, will help to determine what most of us believe and, accordingly, what it means to act on our beliefs. This doesn't apply only to markets, but to any system, political, economic, biological, or otherwise, that's made legible through a deliberately crafted interface.

<meta name="twitter:image:src" content="https://noncomputable.github.io/illustration.png">

{% include js.html %}