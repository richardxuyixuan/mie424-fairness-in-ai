# Optimizing Fairness and Accuracy in Machine Learning

This repository provides the codes for the experiments run in our report "Optimizing Fairness and Accuracy in Machine Learning", which is available at: https://docs.google.com/document/d/1OnRa70_h7VWR_dPjk4JPrW1Edr7V25HF_PKRxT6mksI/edit?usp=sharing. 

Specifically, the code is modified based on the original repository "Fairness in Classification", with is available at: https://github.com/mbilalzafar/fair-classification, and the original repository "SearchFair", which is available at: https://github.com/mlohaus/SearchFair.

In this project, we investigate the impact of incorporating fairness metrics on the accuracy of machine learning models. The goal of these algorithms is to increase fairness without sacrificing accuracy. The first focuses on optimizing demographic parity, and equality of opportunity. The second focuses on optimizing disparate mistreatment. We aim to: 
1. validate these techniques by evaluating the algorithms on two unseen datasets (MEPS and Bank Marketing) that contain protected or sensitive classes (race, age, sex) and are inherently biased towards one protected class over the other, to ensure that the algorithms are robust enough to make accurate and fair predictions in more real-life scenarios.
2. investigate whether optimizing one fairness metric can indirectly improve other fairness metrics. 

Introducing fairness in machine learning is important because industries like the financial sector, job hiring, and the judicial system are using machine learning algorithms to make decision-making processes easier. However, the data used to train the algorithms consists of past decisions made by humans. In these industries, the decisions made in the past have been biased against visible minorities. Historically, African-Americans have been denied mortgage loans from banks more frequently or were given higher mortgage interest rates. In fintech companies, it was found that while machine learning algorithms did reduce bias, they still disproportionately gave African-American clients higher interest rates. In the judicial system, machine learning is being used for risk assessments which are then used to determine sentencing. These algorithms would incorrectly predict that black prisoners were more likely to re-offend, which resulted in longer prison sentences for black offenders. When algorithms are trained on these decisions, it reproduces and reinforces the biases the A.I. was intended to eliminate. Therefore, there is a real motivation, to prioritize fairness as the decisions made by these algorithms have real-life consequences for millions of people around the world.


#### Dependencies 
1. numpy
2. scipy
3. matplotlib
4. CVXPY
5. DCCP

#### Using the code

1. Download the MEPS dataset using this link: https://drive.google.com/file/d/1od1mnkk5NUX80uzXwHnGa9FtpKznEFUa/view?usp=sharing, and insert it into the following directory: disparate_mistreatment/propublica_compas_data_demo
2. Run the following: python disparate_mistreatment/propublica_compas_data_demo/demo_constraints.py
3. Type in the name of the dataset that you would like the algorithms to test on: "bank_marketing", "meps", or "compas".


#### References
1. <a href="http://arxiv.org/abs/1507.05259" target="_blank">Fairness Constraints: Mechanisms for Fair Classification</a> <br>
Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rodriguez, Krishna P. Gummadi. <br>
20th International Conference on Artificial Intelligence and Statistics (AISTATS), Fort Lauderdale, FL, April 2017.
 
 
2. <a href="https://arxiv.org/abs/1610.08452" target="_blank">Fairness Beyond Disparate Treatment & Disparate Impact: Learning Classification without Disparate Mistreatment</a> <br>
Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rodriguez, Krishna P. Gummadi. <br>
26th International World Wide Web Conference (WWW), Perth, Australia, April 2017.