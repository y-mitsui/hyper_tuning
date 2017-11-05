import numpy as np

class ThompsonNormal:
    def __init__(self, n_elements, mean=None, var=None):
        self.n_elements = n_elements
        self.sample = [list() for _ in range(n_elements)]
        self.mean = np.array(mean) if mean is not None else np.array([0.] * n_elements)
        self.var = np.array(var) if var is not None else np.array([100.] * n_elements)
        
    def getPosterior(self):
        sample = []
        n_sample = []
        for row in self.sample:
            sample.extend(row)
            n_sample.append(len(row))
        mother_var = np.var(sample)
        temp = n_sample * self.var
        means = np.array(map(lambda x: np.average(x), self.sample))
        post_mean = (temp * means + mother_var * means) / (temp + mother_var)
        temp2 = np.maximum(mother_var * self.var, 1e-6)
        post_var = temp2 / (n_sample * self.var + mother_var)
        post_var[post_var != post_var] = self.var[post_var != post_var]
        post_mean[post_mean != post_mean] = self.mean[post_mean != post_mean]
        return post_mean, post_var
        
    def get(self):
        post_mean, post_var = self.getPosterior()
        return np.argmax(np.random.normal(post_mean, np.sqrt(post_var)))
        
    def set(self, idx, val):
        self.sample[idx].append(val)
        

if __name__ == "__main__":
    n_elements = 10
    thomson_normal = ThompsonNormal(n_elements, [0.] * n_elements, [100.] * n_elements)
    sample = np.array([thomson_normal.get() for _ in range(1000)])
    print("before", float(sample[sample==0].shape[0]) / sample.shape[0])
    
    for i in range(5000):
        thomson_normal.set(0, np.random.normal(14, 2))
        for j in range(1, n_elements):
            thomson_normal.set(j, np.random.normal(7, 0.3))
        if i % 10 == 0:
            sample = np.array([thomson_normal.get() for _ in range(100)])
            print("success sample after%d"%(i), float(sample[sample==0].shape[0]) / sample.shape[0])
