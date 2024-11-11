
#Original test prompt with answer 

```
    eval_prompt = """You are an OpenMP High Performance Computing expert. Your job is to answer questions about performance optimization. You are given a question and code snippet regarding optimizating the performance.

    You must output the answer to the given question.
    ### Input:
    What is the performance issue in the given code snippet?

    ### Code Snippet:
    #pragma omp parallel shared (a, b) private (c,d) \n { ... \n  #pragma omp critical \n { a += 2 * c; c = d * d; }}

    ### Response: 
    Without the critical region, the first statement here leads to a data race. The second statement however involves private data only and unnecessarily increases the time taken to execute this construct
"""
```

##Base model response to test prompt 

```
    ### Input:
    What is the performance issue in the given code snippet?

    ### Code Snippet:
    #pragma omp parallel shared (a, b) private (c,d) 
     { ... 
      #pragma omp critical 
     { a += 2 * c; c = d * d; }}

    ### Response:
    The performance issue in the given code snippet is that the critical section is too large.
```

##Finetuned  model response to test prompt 

Troubleshooting:

finetuned model is being loaded correctly
no response is being output. 
