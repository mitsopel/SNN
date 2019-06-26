# Mathematics
import numpy as np
import math

EPSILON = 0.0001

# Cost functions

def cross_entropy_error(PredictVector, TargetVector):
    Logits = np.log(PredictVector)
    Values = np.dot(TargetVector, Logits);

    return  ( np.sum(Values) )

# Further explained in the report
def cross_entropy_delta(PredictionVector, TargetVector):
    return (TargetVector - PredictionVector)

# Softmax
def softmax(Vector):
    return [np.exp(Value) / np.sum( np.exp(Vector) ) for Value in (Vector)]

# Partial Derivative of Softmax with specific Index
def softmax_at(Vector, At):
    return np.exp(Vector[At]) / np.sum( np.exp(Vector) )

# Softmax derivative just in case
def softmax_derivative_over(Vector, Index):
    Derivatives = np.empty([len(Vector)])
    for DIndex in range( 0, len(Vector) ) :
        if ( DIndex == Index ):
            Derivatives[DIndex] = softmax_at(Vector, DIndex) * (1 - softmax_at(Vector, Index))
        else:
            Derivatives[DIndex] = -softmax_at(Vector, DIndex) * softmax_at(Vector, Index)

    return Derivatives

# Cosine function

def cos(x):
    return np.cos(x)

def cos_derivative(x):
    return -np.sin(x)

# Softplus function
def softplus( X ):
    return np.log( 1 + np.exp( X ) )

def softplus_derivative( X ):
    return 1 / ( 1+np.exp( -X ) )

# Hyperbolic Tangent
def tanh( X ):
    return np.tanh( X )

def tanh_derivative( X ):
    Tanh = tanh(X);
    return 1 - Tanh * Tanh

# we include the relu for completeness
def relu(x):
#   if x > EPSILON:
    if x > EPSILON:
        return x
    else:
        return 0

def relu_derivative(x):
#   if x > 0
    if x > EPSILON:
        return 1
    else:
        return 0

# we include the sigmoid for completeness
def sigmoid( X ):
    return 1 / ( 1 + np.exp( -X ) )

def sigmoid_derivative(x):
    Sigmoid = sigmoid(x)
    return Sigmoid*(1-Sigmoid)

# Method for gradient checking
# as given in Slide 20 of lec4_b.pdf
# Takes a floating point range
def grad_check(Func, DFunc, Range):
    for x in Range:
        print("Limit Derivative for ", x, " = ", limit_derivative(Func, x), " Analytic Derivative = ", DFunc(x))
        print("Diff for ", x, " = ", np.abs( limit_derivative(Func, x) - DFunc(x) ) )

def limit_derivative(Func, x):
    return ( Func(x + EPSILON) - Func(x - EPSILON) ) / ( EPSILON * 2.0 )

# Normalize to [-1,1] Using Max Value
def normalize(Input, Max):
    Half = Max / 2.0;
#    return ( Input - Half ) / Half
    return ( Input ) / Max

def normalize_stddiv(x, sdiv):
    return x / xmax

def grad_check_all():
    print("RELU Gradient check")
    grad_check(relu, relu_derivative, np.linspace(-5,5,101))

    print("Tanh Gradient check")
    grad_check(tanh, tanh_derivative, np.linspace(-5,5,101))

    print("Sigmoid Gradient check")
    grad_check(sigmoid, sigmoid_derivative, np.linspace(-5,5,101))

    print("Cos Gradient check")
    grad_check(cos, cos_derivative, np.linspace(-5,5,101))
