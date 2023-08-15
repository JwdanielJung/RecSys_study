import numpy as np

class MF():
    def __init__(self, R, K, lr, reg_para, epochs, verbose = False): 
        """
        R: rating
        K: latent  dimension para
        lr: learning rate = alpha on weight update
        reg_para: beta on weight update (regulation term coefficient)
        verbose: 상세한 로깅 출력여부  0 = 출력x, 1 = 자세히, 2 = 함축적 정보
        """
        self._R = R
        self._num_users, self._num_items = R.shape # R = U x I (rating matrix = user matrix x item matrix)
        self._k = K
        self._lr = lr
        self._reg_para = reg_para
        self._epochs = epochs
        self._verbose = verbose

    def fit(self):
        # init user & item latent features
        self._U = np.random.normal(size = (self._num_users, self._k))
        self._I = np.random.normal(size = (self._num_items, self._k))
        
        """
        [논문 기준] 
        global bias = ovarall rating average (total mean) + user bias + item bias
        [코드 기준]
        rating 존재하는 overall rating average 만 global bias로 사용
        """
        # init user & item & global bias
        self._b_U = np.zeros(self._num_users)
        self._b_I = np.zeros(self._num_items)
        self._b = np.mean(self._R[np.where(self._R != 0)]) # global bias /  np.where: 조건 충족 index 찾기

        # training
        self._training_process = []

        for epoch in range(self._epochs):
            for i in range(self._num_users):
                for j in range(self._num_items):
                    if self._R[i, j] > 0:
                        self.gradient_descent(i, j, self._R[i,j])
            cost = self.cost()
            self._training_process.append((epoch, cost))
        # print status
        if self._verbose ==  True and ((epoch + 1) % 10 == 0):
            print(f"Iteration: {epoch + 1} ; cost = {cost:.4f}")

    def cost(self):
        # RMSE cost return
        xi, yi = self._R.nonzero() # 값이 0이 아닌 index 반환
        pred = self.get_complete_matrix()
        cost = 0
        for x,y in zip(xi, yi): # zip 함수 통해 tuple 형태로 만들어서 반환
            cost += pow(self._R[x, y] - pred[x,y], 2)
        return np.sqrt(cost / len(xi)) # RMSE = root(sum {(y - y_pred)^2} / n)
    
    def gradient(self, pred_error, i, j):
        # prediction error (pred_error) = rating - pred
        # user i & item j (index)
        grad_u = pred_error*self._I[j,:] - self._reg_para*self._U[i,:]
        grad_i = pred_error*self._U[i,:] - self._reg_para*self._I[j,:]
        return grad_u, grad_i
    
    def gradient_descent(self, i, j, rating):
        pred = self.get_prediction(i, j)
        pred_error = rating - pred

        # update biases -> 이 부분 왜 이렇게 구현된 것인지 잘 이해안됨,,
        # user i & item j index
        self._b_U[i] += self._lr * (pred_error - self._reg_para * self._b_U[i])
        self._b_I[j] += self._lr * (pred_error - self._reg_para * self._b_I[j])

        # update latent feature
        grad_u, grad_i = self.gradient(pred_error, i, j)
        self._U[i,:] += self._lr * grad_u
        self._I[j,:] += self._lr * grad_i

    def get_prediction(self, i, j):
        # user i & item j index
        # prediction =  overall avg (total bias) + user bias + item bias + latent user * latent item
        return self._b + self._b_U[i] + self._b_I[j] + self._U[i,:].dot(self._I[j,:].T)
    
    def get_complete_matrix(self):
        return self._b + self._b_U[:, np.newaxis] + self._b_I[np.newaxis,:] + self._U.dot(self._I.T)
    

    def print_results(self):
        """
        print fit results
        """

        print("User Latent U:")
        print(self._U)
        print("Item Latent I:")
        print(self._I.T)
        print("U x I:")
        print(self._U.dot(self._I.T))
        print("bias:")
        print(self._b)
        print("User Latent bias:")
        print(self._b_U)
        print("Item Latent bias:")
        print(self._b_I)
        print("Final R matrix:")
        print(self.get_complete_matrix())
        print("Final RMSE:")
        print(self._training_process[self._epochs-1][1])


# run example
if __name__ == "__main__":
    # rating matrix - User X Item : (7 X 5)
    R = np.array([
        [1, 0, 0, 1, 3],
        [2, 0, 3, 1, 1],
        [1, 2, 0, 5, 0],
        [1, 0, 0, 4, 4],
        [2, 1, 5, 4, 0],
        [5, 1, 5, 4, 0],
        [0, 0, 0, 1, 0],
    ])

    # P, Q is (7 X k), (k X 5) matrix
    factorizer = MF(R, K=3, lr=0.01, reg_para=0.01, epochs=300, verbose=True)
    factorizer.fit()
    factorizer.print_results()