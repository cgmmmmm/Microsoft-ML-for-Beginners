# Chapter 1.6: Techniques for Machine Learning (Machine Learning Lifecycle)

Building a ML Model from scratch and then training it is a complex process that takes time and resources. Because of this, before writing any code, we must first assess whether Machine Learning is actually the right approach for our specific problem, or just simply using traditional, rules-based programs would work better.

In this chapter, you'll learn the end-to-end lifecycle of a Machine Learning project:
* **Problem Framing**: Deciding whether AI is the right approach for your problem.
* **Data Preparation**: Collecting, cleaning, and preparing data.
* **Training**: Training/Teaching your model to recognize patterns.
* **Evaluation**: Evaluating how well your model learned.
* **Hyperparameter Tuning**: Tuning the hyperparameters of your model to improve model's performance.
* **Deployment**: Testing the trained model in the real-world.

<br>

---

### 1. Problem Framing

Traditional software is well suited to solve problems where the solution can be described as a formal set of rules.

In contrast, AI shines in solving problems where the solution can be extracted from data.

Many of the problems we encountered in our daily lives can be efficiently solved with traditional programming. If an engineer can break up the solution of a problem, and define it using precise rules, then traditional programming is a great tool to use.

But, many of the problems in our day-to-day aren't quite easy to define as a set of rules.

Thankfully, for many of those problems, we have access to plenty of real life data containing useful information, which means that AI can help us find a solution.

So the first step when starting a new project should be to **analyze the problem (Step 1)**, and determine which technique is best to solve it. If you're able to obtain plenty of data that contains useful information about your solution, then AI is a promising approach.

<br>

---

### 2. Data Preparation

Once you have decided AI is the right method for your problem, you need to **collect and prepare data (Step 2)** before any training begins.

An ML model is only as good as the data it learns from. If you feed it garbage, it will output garbage.

This process is generally broken down into a few critical phases:
* **Data Collection**: First, you must gather your raw data. This could come from databases, spreadsheets, web scraping, or real-time sensors. At this stage, the data is usually messy and unorganized.

* **Data Cleaning**: Real-world data is rarely perfect. You will frequently encounter errors that need to be fixed:
    1. Missing Values
        * If a row is missing a crucial piece of information (like a blank cell in a spreadsheet), you must decide whether to remove that row entirely or fill in the blank with an educated guess (like the average value of that column or something).
    2. Duplicates and Outliers
        * You will need to remove duplicate entries and investigate extreme outliers that could confuse your model.

* **Data Transformation (Preprocessing)**: Machine Learning models are essentially mathematical functions; they only understand numbers.
    1. Encoding
        * If your data includes text categories (like "Red", "Green", or "Blue"), you must convert these into numerical formats.
    2. Normalization
        * If you have data with vastly different scales (e.g., comparing a person's age to their yearly income), you will need to scale to normalize the numbers so they fall into a smaller range. This prevents larger numbers from unfairly dominating the model's learning process.

* **Feature Selection and Engineering**: In ML, the inputs you feed into the model are called *features*, and the output you want to predict is called the *target* or *label*.
    1. You must carefully decide which features are relevant. (e.g., predicting house prices might require the feature "square footage", but "color of the front door" might be irrelevant and should be discarded).
    2. Sometimes, you might even combine existing data to engineer entirely new features that help the model learn better.

* **Data Splitting**: Finally, before moving to the next step (training), you must split your clean data into distinct sets. A large chunk (usually 70-80%) becomes your *Training Data*, and the remainder is hidden away as *Test Data* to evaluate your model later.

<br>

---

### 3. Training 

Now that your data is clean, formatted, and split, you are ready to actually **train your model (Step 3)**

Training a model may take a while, especially if the model is large or if you are feeding it to a massive amount of data. This phase is where the "learning" in Machine Learning actually happens.

During this step, you are essentially teaching a mathematical algorithm to recognize patterns in your data so it can make accurate predictions later.

The training phase can be broken down into these core concepts:
* **Algorithm Selection**: Before the learning begins, you must choose a specific mathematical algorithm (the "blank brain" of your model) that fits your prediction. For example, if you are predicting a number like house price, you might use an algorithm called *Linear Regression*. If you are categorizing emails as spam or not spam, you might use a *Decision Tree*.

* **Feeding and Training Data**: You take the *Training Data* (that 70-80% chunk you set aside in Step 2) and pass it through your chosen algorithm.

* **The Learning Loop (Iterative Process)**: The model doesn't learn everything instantly; it learns through trial and error. This happens in a continuous loop:

    1. **Guess**: The model looks at the input features (e.g., house size, number of bedrooms) and makes a blind guess at the target (e.g., the house price).

    2. **Check**: The model compares its guess to the actual correct answer provided in your training data to see how far off it was. This difference is called the *error* or *loss*. In Machine Learning, a **Loss/Error Function** is a mathematical formula that quantifies the difference between the actual target value (y) and the predictive guess (ŷ) made by a model. A **Cost Function** averages or sums the loss function over the entire dataset.

    3. **Adjust**: The model uses complex math behind the scenes to tweak its internal settings (called *parameters* or *weights*). It adjusts these settings slightly so that its next guess will be a little closer to the correct answer.

* **Time and Computing Power**: This guess-check-adjust loop happens thousands or even millions or times. For simple problems, training might take a few seconds on a standard laptop. For a highly complex problems, like training large AI language models or image recognition systems, training can take weeks and require massive, specialized computer processors called GPUs.

When we train a model, we are using historical data. Because these events have already happened, we know the final outcome (the correct answer). We use this historical data as an "answer key" to teach the model how the world works.

<br>

Think of training an ML model like a student preparing for a math exam:

1. **Training**: The student takes a practice test and uses the answer key at the back of the book to check their work. If they got the answer wrong, they look at the correct answer, figure out where they made a mistake, and adjust their thinking. (This is the **guess-check-adjust** loop).

2. **Predicting**: When the student sits down for the actual exam, there is no answer key. They have to use the patterns and rules they learned from the practice test to solve brand new problems.

<br>

Real-World Example: Predicting House Prices

1. **The Training Data (The Past)**: We have a dataset of 10,000 houses that were sold last year. Because they're already sold, we know their exact square footage, number of bedrooms, and their final sale price (the correct answer). We use this to train the model.

2. **The Goal (The Future)**: You want to sell your house tomorrow. You know the square footage and bedrooms, but you don't know what the price should be. You feed your house's features into the trained model, and it uses what it learned from last year's data to predict a price for you.

<br>

---

### 4. Evaluation

Once the model is trained, you must **(Step 4) evaluate** its performance using the remainder *Test Data* that you set aside during the Data Preparation phase.

It's critical that you test the algorithm with data that it hasn't seen before or during training, to ensure that it generalizes well to new scenarios.

If you test a model using the exact same data that it was trained on, it is like giving the student the final exam ahead of time. They might just memorize the answers instead of actually learning the concepts.

The evaluation phase usually involves a few key concepts:

1. **Running the Test Data**: You pass your hidden *Test Data* (the remaining 20-30% of your clean data) through the newly trained model, asking it to make predictions. You then compare the model's predictions against the actual, correct answers in the test data to see how well it performed.

2. **Measuring Success (Metrics)**: You need a mathematical way to score the model's "exam". Depending on your specific problem, you will use different evaluation metrics:

    * For categorizing **(Classification)**: If you are predicting categories (like "Spam" vs "Not Spam"), you might look at *Accuracy* (what percentage of guesses were exactly right?).

    * For predicting numbers **(Regression)**: If you are predicting a continuous number (like house prices), you might look at the *Mean Absolute Error* (on average, how far off was the model's prediction from the actual price?).

3. **Diagnosing the model**: The evaluation score helps you identify two of the most common problems in Machine Learning:

    * **Overfitting**: The model scores perfectly on the training data, but performs terribly on the test data. This means it essentially "memorized" the training data's exact quirks and noise, but failed to learn the underlying general patterns.

    * **Underfitting**: The model performs poorly on both the training data and the test data. This means the algorithm was likely too simple, or wasn't trained long enough, and failed to capture any meaningful patterns at all.

<br>

---

### 5. Hyperparameter Tuning

If your evaluation reveals that your model isn't performing as well as you'd like (for example, it is overfitting or underfitting), you don't necessarily need to throw it away. Instead, you can improve it by **adjusting its hyperparameters (Step 5)**.

Some algorithms contain hyperparameters, which are settings that control key aspects of their inner workings. Choosing good hyperparameters is important because they can make a big difference in your results.

To understand hyperparameters, think about baking a cake. The ingredients you use are your *Training Data*. The chemical reactions that happen inside the cake as it bakes are the *parameters* (the patterns the model learns on its own). But the oven's temperature and the baking time are the *hyperparameters*, they are external settings you must choose before the baking even begins.

If you want to be systematic about your hyperparameter search, you can write code that tries lots of different combinations and helps you discover the best values for your data.

Choosing good hyperparameters is critical because they control the inner workings of your algorithm and can drastically change your results. Here is how this tuning process works:

* **The Control Panel**: Every ML algorithm comes with its own unique set of "dials" or hyperparameters.
    * For example, a hyperparameter might control the *Learning Rate* (how aggressively the model changes its guesses after making a mistake) or the *Maximum Depth* (how complex you allow a decision-making algorithm to get).

* **Fixing Mistakes**: If your model is *overfitting* (memorizing the data), you can adjust a hyperparameter to force the algorithm to be simpler. If it is *underfitting* (failing to learn), you can turn a dial that allows the algorithm to look for more complex patterns.

* **Automating the Search**: You rarely know the perfect settings right away. Instead of guessing manually, data scientists write code to systematically test different combinations.

    * **Grid Search**: You give the computer a list of possible settings, and it exhaustively tests every single combination to ind the absolute best one.

    * **Random Search**: Because Grid Search can take a massive amount of time, you can instead tell the computer to test a random sampling of combinations, which often finds a "good enough" setting much faster.

<br>

---

### 6. Deployment

Once you get good test results and you have tuned your parameters, it's time to see how well your model performs within the context of its intended use. **Deployment (Step 6)** is the process of taking your trained machine learning model and making it available to real users or other software systems.

This could involve collecting live data from a sensor and using it to make predictions, or deploying a model to a few users of your application.

This phase takes the model out of the lab and puts it to work. There are some key considerations when deploying a model:

1. **Integration**
    * A trained model is essentially a file containing mathematical rules. To be useful, it must be integrated into a larger traditional software system. This usually involves wrappingt he model in an API so that a website, a mobile app, or a live sensor can send it new data and instantly receive a prediction back.

2. **Phased Rollouts**
    * You rarely want to release a brand new model to 100% of your users all at once. Instead you might use a strategy like A/B Testing:
        * You deploy the new ML model to small percentage of your application's users.
        * You keep the rest of the users on the old system.
        * You compare the results to ensure the new AI model is actually providing a better user experience or more accurate results before a full release.

3. **Monitoring for "Drift"**
    * The real world is constantly changing, which means the data your model sees in production will eventually change, too.
    * For example, a model trained to predict clothing trends in 2019 would perform terribly in 2020. This degradation is called *Data Drift*. You must continuously monitor your model's real-world accuracy to ensure it hasn't become outdated. 

4. **The Continuous Loop**
    * Because of Data Drift, the Machine Learning lifecycle is never truly finished; it is a continuous loop. Once a deployed model's performance starts to drop, it is time to start the cycle all over again: you collect new, up-to-date data, retain the model, evaluate it, and deploy an updated version.

<br>

---

### 🔴 This marks the end of Chapter 1 of the Microsoft ML for Beginners Course. 🔴
Chapter 2 will go over theoretical knowledge and practical skills about **Regression**.