# POLICY ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.

## PROBLEM STATEMENT
The aim of this experiment is to find optimal policy for the mdp using policy iteration. Policy iteration includes policy evaluation and policy improvement where evaluation function is used to find optimal value function of each state and then improvement function is used to find best policy by comparing all the action value function as well as policy.

## POLICY ITERATION ALGORITHM
-> Step1 :
we are going to do policy evaluation of each state to get the state value function where the initial policy is defined randomly to the mdp.

-> Step2:
Once we obtain convergence in the policy evaluation then implement policy improvement where we are going to find best optimal policy until the previous and current policy are same.
</br>
</br>


## POLICY IMPROVEMENT FUNCTION
### Name : Santhana Lakshmi K
### Register Number : 212222240091
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    # Write your code here to improve the given policy
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob,next_state,reward,done in P[s][a]:
          Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
          new_pi=lambda s:{s:a for s, a in enumerate(np.argmax(Q,axis=1))}[s]
    return new_pi
```
## POLICY ITERATION FUNCTION
### Name : Santhana Lakshmi K
### Register Number :212222240091
```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
   random_actions=np.random.choice(tuple(P[0].keys()),len(P))
   pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
   while True:
    old_pi = {s:pi(s) for s in range(len(P))}
    V = policy_evaluation(pi, P,gamma,theta)
    pi = policy_improvement(V,P,gamma)
    if old_pi == {s:pi(s) for s in range(len(P))}:
      break
   return V, pi
```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
</br>![image](https://github.com/user-attachments/assets/a01e35f4-a5b0-4c92-a217-4bffe4bd542f)

</br>![image](https://github.com/user-attachments/assets/5891a8fa-630a-426c-b0f1-1e1442cba5a8)


### 2. Policy, Value function and success rate for the Improved Policy
</br>![image](https://github.com/user-attachments/assets/46411244-7401-4ada-b752-2d98be60ce60)

</br>![image](https://github.com/user-attachments/assets/c5ef5c5b-cdee-44b2-a835-7e232205839e)


### 3. Policy, Value function and success rate after policy iteration
</br>![image](https://github.com/user-attachments/assets/b3499e1b-66fc-4b76-a08f-2f72a7c3358f)

</br>![image](https://github.com/user-attachments/assets/33e445ed-3288-4dd6-b13a-6f64cc873518)



## RESULT:

Thus, the program to iterate Policy improvement and evaluation is implemented successfully

