import numpy as np

def cal_weights(vectors):
    num_vectors,vector_length = vectors.shape
    weights = np.dot(vectors.T,vectors)
    np.fill_diagonal(weights,0)
    return weights/num_vectors

def activation(x):
    return np.where(x>=0,1,-1)

def hop(input_vector,weights):
    return activation(np.dot(weights,input_vector))

def main():
    vectors=[]
    for i in range(4):
        vector_str = input(f"enter vector{i+1}(comma seperated:)")
        vector = [int(x) for x in vector_str.split(',')]
        vectors.append(vector)
    vectors=np.array(vectors)

    weights = cal_weights(vectors)
    input_vector = vectors[0]
    output_vector = hop(input_vector,weights)

    print(input_vector)
    print(output_vector)

if __name__=="__main__":
    main()
