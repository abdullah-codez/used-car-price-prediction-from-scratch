import numpy as np

def predict(X_train, w, b):
    return np.dot(X_train, w) + b
    
def cost_function(m, w, b, X_train, y_train):
    y_pred = predict(X_train, w, b)
    error = y_pred - y_train
    return (1/(2*m))*np.sum(error**2)
    
def compute_dw_db(m, w, b, X_train, y_train):
    y_pred = predict(X_train, w, b)
    error = y_pred - y_train
    dw = (1/m)*np.dot(X_train.T, error) 
    db = (1/m)*np.sum(error)
    return dw, db
    
def gradient_descent(w, b, X_train, y_train, epochs, alpha):
    cost_history = []
    for i in range(epochs):
        dw, db = compute_dw_db(m, w, b, X_train, y_train)
        w = w - alpha *dw
        b = b - alpha *db
        cost = cost_function(m, w, b, X_train, y_train)
        cost_history.append(cost)
        if i % 100 == 0:
            print(f" iteration {i} : Cost : {cost}")

    return w , b , cost_history

alpha = 0.01
epochs = 10000
w_final, b_final, cost_history = gradient_descent(w, b, X_train, y_train, epochs, alpha)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    