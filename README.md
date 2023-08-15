**Linear Regression**

**Dataset.** In this programming homework, we will use a Turbo.az dataset which is a collection of cars (Mercedes C-class). It contains information about 1328 cars with features of (Sheher, Marka, Model, Buraxilish ili, Ban novu, Reng, Muherrikin hecmi, Muherrikin gucu, Yanacaq novu, Yurush, Suretler qutusu, Oturucu, Yeni, Qiymet, Extra Information, Seller’s comment).

**1. Loading data (10%).** Using pandas library in python is recommended.

You will need to read the data from data file (turboaz.csv) and extract only 3 columns for your model:

X<sub>1</sub> = “Yurush” (Milage). If samples of “Yurush” are described by string format, remove “km” from the string and convert it to the number format.

X<sub>2</sub> = “Buraxilish ili” (Model Year).

Y = “Qiymet” (Price). If prices of car are given in dollar ($) convert them to manat (AZN).

**2. Visualization (10%).** Using matplotlib library (scatter, Axes3D) in python is recommended.

You will need to provide 3 visualizations of data.

<ol type="a">
   <li>
      Qiymet (Y) vs Yurush (X<sub>1</sub>)
   </li>
   <li>
      Qiymet (Y) vs Buraxilish ili (X<sub>2</sub>)
   </li>
   <li>
      3D plot of all three values (Y, X<sub>1</sub>, X<sub>2</sub>)
   </li>
</ol>

**3. Implementation of Linear Regression from scratch (40%)**

<ol type="a">
   <li>
      Calculate Cost function. Implement a function which returns cost given true y values, x values and coefficients (w).
   </li>
   <li>
      Normalize data using Z score normalization (Recommended).
   </li>
   <li>
      Implement gradient descent algorithm to minimize Cost Function.
      <ul>
         <li>
            Assign initial values of W=(w<sub>0,</sub> w<sub>1</sub>, w<sub>2</sub>) to zero or choose randomly.
         </li>
         <li>
            Learning rate: alpha=0.001, you can change it in different experiments.
         </li>
         <li>
            Number of iterations: 10000 or you can stop it when two sequential values are too close.
         </li>
         <li>
            Calculate values of parameters using Gradient descent formula.
         </li>
      </ul>
   </li>
   <li>
      Plot graph of Cost function and describe how it changes over iterations.
   </li>
   <li>
      Plot points of Y (Qiymet) vs X1 (Buraxilish ili) and draw a line of predictions made with parameters you got from gradient descent.
   </li>
   <li>
      Plot points of Y (Qiymet) vs X<sub>2</sub> (Yurush) and draw a line of predictions made with parameters you got from gradient descent.
   </li>
   <li>
      Plot 3D graph of points of Y (Qiymet), X<sub>1</sub>, X<sub>2</sub> and predicted Y (Qiymet) using the same X<sub>1</sub> and X<sub>2</sub>. It should look like this (blue points are true values, red points are predicted values):
   </li>
   <li>
      <strong>Testing.</strong>
      <p>
         Here are given two new cars which are not in the dataset:
      </p>
      <p>
         Car 1 { Yurush: 240000, Buraxilish ili: 2000, Qiymet: 11500}
      </p>
      <p>
         Car 2 { Yurush: 415558, Buraxilish ili: 1996, Qiymet: 8800}
      </p>
      <p>
         Predict prices of these cars by using your parameters and compare with actual prices.
      </p>
   </li>
</ol>

**4. Linear Regression using library (20%).**

Use a library to fit Linear Regression to the data. You should use the same features (Yurush and Buraxilish ili) as input to this model and perform the same testing (3(h)) as above. (Using scikit-learn library is recommended).

**5. Report (20%).** Write codes, their explanation and achieved results for each step (1-4) of homework in the report. Codes must be in text format. You can take screen shots only for output of program.

**Extra tasks:**

1. Solve linear regression by Normal equation (10%)
