# Chapter 1.5: Minkowski Metric and Feature Vector Distance

This chapter will involve the example used in Chapter 1.4. 

After conducting **Feature Engineering** in the our theoretical ML Model of deducing whether an animal is a reptile or not, the current model deduces that a reptile:
* Has scales
* Cold-blooded

Well, a salmon has scales and is also cold-blooded, but it isn't a reptile. But that's okay, no model is perfect, as the saying goes, 
> ***"All models are wrong, but some are useful."***. **- George Box**

Trying to achieve perfection will lead to **overfitting**, making the model worse.


This chapter will focus on **Minkowski Distance**; Used to compare features, figure out how to group a certain demographic together, or how to find a dividing line that separate things apart.

<br>

---

### Minkowski Distance

<br>

What is "Minkowski Distance"?

Minkowski Distance in machine learning, is the proxy for *similarity*; if we can define distance, we can define how "alike" two pieces of data are.

* Formula of Minkowski Space's Distance:

$$
\begin{gathered}
{\Large d(X, Y, p) = \left( \sum_{i=1}^n | x_i-y_i|^p \right)^{\frac{1}{p}} } \\
\\
\text{where:} \\
p = 1 \text{ for Manhattan distance (L1 Norm)} \\
p = 2 \text{ for Euclidean distance (L2 Norm)}
\end{gathered}
$$

* Manhattan Distance:
    * Used for **high-dimensional data**, as the number of dimensions increases *("The Curse of Dimensionality")*, Manhattan Distance often becomes more stable and meaningful than Euclidean.
    * **Use-case**: in scenarios where movement is constrained to a grid, navigation & logistic systems, image processing, robust regression.
    * **Characteristics**: Follows a "grid" path. It is less sensitive to outliers than Euclidean Distance.
    * **Outliers**: If data contains extreme values, Manhattan is more robust to outliers as it does not square the differences, preventing large deviation from dominating the calculation. 

* Euclidean Distance:
    * Used for **low-dimensional clustering**, when features are continuous and dense.
    * **Use-case**: K-means, KNN, Image Recognition
    * **Characteristics**: The "straight-line" distance. Most common for physical measurements and general clustering.
    * **Normally Distributed Data**: It works best when continuous features follow a Gaussian (normal) distribution.

<br>

By learning this formula and adjusting parameter *`p`*, you can gain access to a wide range of behaviours for your models.

<br>

---

### Example from Chapter 1.4

One way to separate out reptiles from non-reptiles is to measure the distance (using ***Minkowski Space's Distance***) between pairs of examples, and use that to decide what's near each other.

There is five binary features and one integer feature associated with each animal.

<p align="center">
  <img src="assets/python-data.png" alt="Animal Data">
</p>

In our final ML model, the model deduces that animal with these **THREE(3)** traits are reptiles:
1. Has Scales
2. Cold-blooded
3. Has 0 or 4 legs

<br>

```python
# convert examples into feature vectors
rattlesnake = [1, 1, 1, 1, 0]
boa_constrictor = [0, 1, 0, 1, 0]
dart_frog = [1, 0, 1, 0, 4]
alligator = [1, 1, 0, 1, 4]

# Minkowski Distance in Python
def minkowski_dist(x, y, p):
    dist = 0
    n = len(x) # or n = len(y)
    for i in range(0, n): # standard python is 0-indexed
        dist += abs(x[i] - y[i])**p
    return (dist)**(1/p)
``` 

* `x` and `y` represents the pair of animals:
    * `x` can be ( `rattlesnake` or `boa_constrictor` or `dart_frog` )
    * `y` can be ( `rattlesnake` or `boa_constrictor` or `dart_frog` )

* `p` represents the value determined by:
    * Manhattan Distance - *1*
    * Euclidean Distance - *2*

#### NOTE: The smaller the value, the greater the similarity, indicating the pair of animals are closely mapped within the multidimensional feature space.

<br>

### Using Euclidean Distance (`p` = 2) 
#### Distance between Rattlesnake and Boa Constrictor
```python
# Computed distance between rattlesnake and boa constrictor
print(minkowski_dist(rattlesnake, boa_constrictor, 2))

Result: 1.41421356
```
What this says: Rattlesnake and Boa Constrictor are reasonably close/similar to each other.

<br>

#### Distance between Rattlesnake and Dart Frog
```python
# Computed distance between rattlesnake and Dart Frog
print(minkowski_dist(rattlesnake, dart_frog, 2))

Result: 4.24264068
```
What this says: Rattlesnake and Dart Frog have less similarities compared to Rattlesnake and Boa Constrictor.

<br>

#### Distance between Boa Constrictor and Dart Frog
```python
# Computed distance between Boa Constrictor and Dart Frog
print(minkowski_dist(boa_constrictor, dart_frog, 2))

Result: 4.47213595
```
What this says: Boa Constrictor and Dart Frog have less similarities compared to Rattlesnake and Boa Constrictor.

<br>

#### Distance between Alligator and the rest of the animals
```python
print(minkowski_dist(alligator, rattlesnake, 2))
Result: 4.12310562

print(minkowski_dist(alligator, boa_constrictor, 2))
Result: 4.12310562

print(minkowski_dist(alligator, dart_frog, 2))
Result: 1.73205080
```
What this says: Alligator is more similar to Dart Frog than the other Rattlesnake and Boa Constrictor.

But this is **incorrect**. 
* An Alligator is a reptile, a Dart Frog is not. The Alligator should be similar to Rattlesnake and Boa Constrictor.

So ***why***?
* Alligator differs from Dart Frog in $3$ features, from Boa in only $2$ features.
* But scale on "legs" is from $0$ to $4$, on other features is $0$ to $1$.
* "legs" dimension is disproportionately large.

<br>

### Table of Values obtained using non-binary features "number of legs"

|                     | Rattlesnake | Boa Constrictor | Dart Frog | Alligator |
|---------------------|-------------|-----------------|-----------|-----------|
| **Rattlesnake**     | -  | 1.41421356      | 4.24264068| 4.12310562|
| **Boa Constrictor** | 1.41421356  | -      | 4.47213595| 4.12310562|
| **Dart Frog**       | 4.24264068  | 4.47213595      | - | 1.73205080|
| **Alligator**       | 4.12310562  | 4.12310562      | 1.73205080| - |

<br>

### How to fix this inaccuracy?

#### Solution: 
> Apply feature scaling (normalization or standardization) before computing distance

> Alternatively, apply feature weighting if domain knowledge suggests certain traits are more important.

> In this simple example: Make every feature binary; whether it *has legs* or it *does not have legs*. 

#### Why ?
* As observed from the feature vectors, the "number of legs" feature was disproportionately weighing the Euclidean distance calculation.
* The number of legs ranged from $0$ to $4$, while other features were binary ($0$ or $1$).
* This made the "number of legs" dimension too large and skewed the distance measurements.

#### Distance between Alligator and the rest of the animals, using Binary Features
```python
rattlesnake = [1, 1, 1, 1, 0]
boa_constrictor = [0, 1, 0, 1, 0]
dart_frog = [1, 0, 1, 0, 1] # updated to use binary features
alligator = [1, 1, 0, 1, 1] # updated to use binary features

# Minkowski Distance in Python
def minkowski_dist(x, y, p):
    dist = 0
    n = len(x)
    for i in range(0, n):
        dist += abs(x[i] - y[i])**p
    return (dist)**(1/p)

print(minkowski_dist(alligator, rattlesnake, 2)) 
Result: 1.41421356 # before: 4.12310562

print(minkowski_dist(alligator, boa_constrictor, 2))
Result: 1.41421356 # before: 4.12310562

print(minkowski_dist(alligator, dart_frog, 2))
Result: 1.73205080
```
What this says: Now, an Alligator is more similar to Rattlesnake and Boa Constrictor than it is similar to Dart Frog. 

<br>

### Table of Values obtained using binary features "number of legs"

|                     | Rattlesnake | Boa Constrictor | Dart Frog | Alligator |
|---------------------|-------------|-----------------|-----------|-----------|
| **Rattlesnake**     | -  | **1.41421356**    | 1.73205080| **1.41421356**|
| **Boa Constrictor** | **1.41421356**  | -    | 2.23606797| **1.41421356**|
| **Dart Frog**       | 1.73205080 | 2.23606797 | - | 1.73205080|
| **Alligator**       | **1.41421356**  | **1.41421356** | 1.73205080| - |

<br>

#### What does all of this mean?
* Choice of features matters. ( Feature Engineering )
* Using too many features may result in ***overfitting***.
* Deciding the weights on those features has a real impact.

<br>

#### NOTE: Using Manhattan Distance AND binary features is more preferred here, I'm not going to run the Manhattan metric. But if I do, the result would also get that Alligator is much closer to the snakes because it differs in only two features, not three. 

<br>

---

### 🔴 This marks the end of Chapter 1.5 of the Microsoft ML for Beginners Course. 🔴
Chapter 1.6 will discuss about **Techniques of Machine Learning**.