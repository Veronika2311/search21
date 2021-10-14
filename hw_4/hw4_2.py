from hw_4_1 import _result_for_metrics_fasttext, _result_for_metrics_bert

#fasttext = 0.0128
true_ans, all_ans = _result_for_metrics_fasttext()
print('score of fasttext')
print(true_ans / all_ans)

#bert = 0.0085
true_ans, all_ans = _result_for_metrics_bert()
print('score of bert')
print(true_ans / all_ans)
