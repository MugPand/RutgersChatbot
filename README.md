# RutgersChatbot

## Abstract
Course planning is a difficult task that students face every semester. Composed of verifying prerequisites, major/minor requirements, and the difficulty of their classes, students often turn to academic advisors with many of their questions. Advisors are often overwhelmed by the quantity of these questions when the majority can be answered through online resources. Our project explores the automation of these advising tasks with a conversational AI system. 

The focus of our research was on creating a generalizable system for automated conversation on advising topics; our tests focused on the Computer Science Major & Minor, and the newly introduced Data Science Certificate & Minor.

## Background

We started with a conversational A.I. system that used matches between inputs and a list of response functions. This system included some simple conversation (“Hello”, “Goodbye”, “How are you? ”), basic course recommendation, and information retrieval. 

Our goal was to extend the capabilities of the chatbot into a more comprehensive user-experience. This required several interdisciplinary approaches including analyzing how questions/answers should be framed, programming technicalities, machine learning strategies, U.I. design, and user testing. 

## System Design

- We have utilized Python, and external libraries such as SpaCy, Scikit Learn, and Pandas to develop a chatbot capable of interacting with users. 
- We we were able to scrape Rutgers websites for course information and build a database using SQL.
- For the user-interface, we worked with the Flask and React frameworks to present the chatbot in an environment similar to its real-world use case.

## Future Direction:
- Broader & more complex questions and responses
- Apply to a broader selection of courses and majors
- Improve transition system & state tracking for conversations

## Reflection
- Generally, the project was an interesting experience to explore human conversation and translate that into a digital medium.
- The concept we are implementing can be applied to a variety of problems; it isn’t dependent on ‘course advising’ in particular.
- Our approach is modular and leaves many spaces where additional functionality can be added, either within our current framework or by modifying it.
- Could the responsibilities of academic advisors in 5-10 years realistically be fulfilled by conversational AI?


## Acknowledgements
We want to thank Dr. Matthew Stone for his assistance on this project throughout the semester and for giving us the opportunity to work on this course planning chatbot!
