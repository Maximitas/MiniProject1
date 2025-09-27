This is the read me for MiniProject3

1) Which are the most decisive factors for quitting a job? Why do people quit their job?
So my classification output showed a 56% accuracy when people left, meaning it was complicated reasons with little data.
My correlatin matrix showed a low corralation with Scientist, Technician, Representative
So low income could be a reason.  

2) Which work positions and departments are in higher risk of losing employees?
My corralation matrix again shows that Scientist, Technician, Representative have low income and low income typically corralate with attrition.

3) Are employees of different gender paid equally in all departments?
I checked the corralation matrix to monthly income and saw that gender was only at -0,03 so they are basically paid the same

4) Do the family status and the distance from work influence the work-life balance?
So, the distance from home shows in my code that there really isnt a diffrence at -0.05
MartalStatus_Single got a bit more but still low at 0.178.
So no it dosnt really effect work life.

5) Does education make people happy (satisfied from the work)?
Nope, it dosnt matter really the education compared to work satisfaction. Its at -0.01

6) Which machine learning methods did you choose to apply in the application and why?
I used Linear regression to predict income.
I also used logistic regression topredict who leaves their job.
I at last used KMeans clustering to find hidden groups in the data.

7) How accurate are your solutions of prediction? Explain the meaning of the quality measures.
My logistic regression had an accuracy of 87%. The recall accuracy was at 95% and the accuracy for people who left was only at 56%.
The linear regression had an avearge fail on the income prediction at 1140. I think it is pretty low when the incomes where from 5000 to 20000.

8) What could be done for further improvement of the accuracy of the models?
Remove more outliers. Try diffrent combinations of viarables. Try with or withour corraltion matrix. 

9) Which were the challenges in the project development?
There was a lot of colls after i did one-hot coding to make the data mahinelearning ready. Some of the data was off, my accuracy for
people leaving was only at 56% procent. On the 4th step of the assignment my best cluster was only on 0.122.

To find about in my code and my data. I could definitely have used more time on structure. But i was low on time as i was sick most of the week
and i was a 1 man army for this assignment. I had to learn alot for the 3rd and 4th step of the assignment as i couldnt be there for the lectures.

