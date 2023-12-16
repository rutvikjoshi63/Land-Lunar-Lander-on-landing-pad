# Deep Dive into Machine Learning Specialization
## Table Of Contents

  * [Introduction](#1.-introduction)
  * [Skills Developed](#skills-developed)
  * [Projects](#getting-started)
    + [Land Lunar Lander on landing pad](#land-lunar-lander-on-landing-pad)
      - [Code Base](#codebase)
      - [Procedure](#procedure)
      - [Learnings](#learnings)

  * [Resources](#resources)

 
# 1. Introduction
Having completed DeepLearning.AI's Machine Learning Specialization, I feel empowered with a robust foundation in this transformative field. Professor Ng's masterful guidance expertly navigated the complexities of supervised and unsupervised learning, neural networks, and reinforcement learning.

The program surpassed mere technical training, delving into industry best practices and ethical considerations, equipping me with a Silicon Valley-caliber approach to AI development. This practical knowledge translates directly into tackling real-world challenges with confidence.

The Specialization provided me with:
# Skills Developed
* Solid understanding of foundational machine learning concepts: From linear regression to deep reinforcement learning, I gained a comprehensive grasp of key algorithms and their applications.
* Hands-on Python skills: Building models using NumPy and scikit-learn solidified my ability to implement these concepts in real-world scenarios.
* Critical thinking and problem-solving: I learned to approach challenges with a data-driven mindset, employing best practices for model evaluation and performance optimization.
* Exposure to cutting-edge advancements: The curriculum covered the latest trends in AI, including recommender systems and deep learning, preparing me for the evolving landscape of the field.

Overall, the Machine Learning Specialization proved to be an invaluable investment, equipping me with the knowledge and skills to embark on a successful career in AI. I am grateful to Professor Ng and the DeepLearning.AI team for crafting such a comprehensive and impactful learning experience.

Looking forward to: Utilizing my newfound expertise to contribute to meaningful AI solutions that address real-world problems and shape the future.

# Resources:

DeepLearning.AI: https://www.deeplearning.ai/
Stanford Online: https://online.stanford.edu/
Andrew Ng: https://www.youtube.com/watch?v=779kvo2dxb4

# Projects
## Land Lunar Lander on landing pad
This project was a hands-on introduction to the world of reinforcement learning, where an agent learns through trial and error, maximizing its reward by making optimal decisions in a dynamic environment.
### [Code Base](https://github.com/rutvikjoshi63/Land-Lunar-Lander-on-landing-pad/tree/main/LunarLander/Files)

+ **Procedure**
  * Modeling the moon environment: I built a physics simulation of the moon's gravity, thrusters, and lander dynamics. This provided the foundation for the agent's decision-making.
  * Crafting a reward function: I defined a reward system that prioritized landing within the designated area at low velocity, penalizing crashes and out-of-bounds landings. This guided the agent's learning towards the desired outcome.
  * Implementing Q-Learning: I used the Q-Learning algorithm, which learns the value of taking different actions in different states. The agent constantly updated its "Q-table," estimating the future rewards associated with each action in a specific situation.
  * Training through exploration and exploitation: The agent initially explored the environment randomly, learning the consequences of each action. As it gained experience, it exploited its knowledge, favoring actions with higher predicted rewards, leading to smoother and safer landings. 
+ **Learnings**
  * Understanding the principles of reinforcement learning: I grasped the core concepts of reward maximization, Q-tables, and the trade-off between exploration and exploitation.
  * Applying theoretical knowledge to a real-world scenario: The project bridged the gap between theory and practice, demonstrating how reinforcement learning can be used to solve complex problems.
  * Developing problem-solving skills: I learned to break down a challenging task into smaller components, define objectives, and iterate on my approach to achieve optimal results.

