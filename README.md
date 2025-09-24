## Overview
The goal of this project is to develop a specialized framework that adpatively and elegantly balances the trade-offs
between personalization, generalization, privacy, and efficiency (Adaptive Personalized Federated Learning) specifically
for genomic data and the task of predicting "polygenic risk score" collaboratively. The framework would be answering three
questions at least for this domain
* The heterogeneity unique to genomic data
  - ancestral diversity
  - technical diversity
1. What type of heterogeneity are we characterizing and how do we diagnose complex heterogeneity under strict privacy and communication
constraints. (Do we need to simulate a data breach to confirm that whatever method is being used to diagnose heterogeneity cannot leak private information)
2. How do we navigate the vast search space of algorithms and hyperparameters efficiently? (Do we limit the algorithms, can we represent heterogeneity in a manner
that makes it easy to identify the personalization algorithm to implement, how do we teach the RL agent how to remember which algorithm worked the best for a heterogeneity class)
3. How do we ensure stable convergence while dynamically adapting the learning strategy?

## Rough direction
- since we are building a specialized adaptive framework, we would need to identify what the specify type of heterogeneity we would be handling
for this data is -> genomic data for PRK. different sequencing technology, ancestry, admixture ...
- Based on the heterogeneity that is observable within our data, we can define personalisation techniques that can work without the risk of training instalbility where strategy might 
change from a robust global model to a clustered approach and then later fine-grained personalisation -> maybe we focus on fine-grained personalisation, 
and note this is a limitation of the framework, and that for certain heterogeneity that could best be naturally handled with client clustering we would be
going for a likely suboptimal solution for instance training genomic data from european descent in a cluster.
- what would be the best way to identify the different heterogeneity and what are the privacy concern and how much of a concern is it? How can we abstract
semi-sensitive data if possible and identify the trade-offs when it isn't.
- can we handle temporal data heterogeneity shift

## Rough system architecture
we have aware-clients containing data, we have the central server containing the global model and a hypernetwork for generating

## Action Items

### Literature Review
- [ ] Search for federated learning + genomic data papers (PubMed, arXiv)
- [ ] Review 10-15 relevant papers on data heterogeneity handling
- [ ] Document key approaches and methodologies

### Data Heterogeneity Analysis  
- [ ] Catalog genomic data heterogeneity types (batch effects, population stratification, missing data)
- [ ] Identify which types are most relevant to our use case
- [ ] Create heterogeneity simulation strategies

### Evaluation Framework
- [ ] Define model performance metrics (accuracy, F1, AUC)
- [ ] Convergence rate
- [ ] Define fairness metrics across different populations/batches (model performing poorly for client with underrepresented data types)
- [ ] Establish statistical significance testing approach



