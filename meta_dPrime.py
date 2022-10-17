from sklearn import metrics
import matplotlib.pyplot as plt

def dPrime(data):

  FP_rate_li = np.array([])
  TP_rate_li = np.array([])

  for subj in data['subject_num'].unique():
    df = data.loc[data['subject_num']==subj]

    TP = df.loc[df['D'] == 'True_positive'].shape[0]
    FN = df.loc[df['D']=='False_negative'].shape[0]

    TN = df.loc[df['D']=='True_negative'].shape[0]
    FP = df.loc[df['D']=='False_positive'].shape[0]

    if TP + FN == 0:
      continue
    TP_rate = TP/(TP + FN)
    TP_rate_li = np.append(TP_rate_li, TP_rate)

    FP_rate = 1 - TP_rate
    FP_rate_li = np.append(FP_rate_li, FP_rate)

  final_FP_rate_li = np.sort(FP_rate_li)
  final_TP_rate_li = np.sort(TP_rate_li)

  return final_FP_rate_li, final_TP_rate_li

fpr, tpr = dPrime(exp_1)
roc_auc = metrics.auc(fpr, tpr)
plt.title("[Block 1] ROC of Meta-d'")
plt.plot(fpr, tpr, 'c', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
