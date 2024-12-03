from collections import Counter,defaultdict
import math

# 训练数据 (表3)
data = [
    ('青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '是'),
    ('乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '是'),
    ('乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '是'),
    ('青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '是'),
    ('浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '是'),
    ('青绿', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.403, 0.237, '是'),
    ('乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '是'),
    ('乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '硬滑', 0.437, 0.211, '是'),
    ('乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '否'),
    ('青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '否'),
    ('浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '否'),
    ('浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '否'),
    ('青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '否'),
    ('浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '否'),
    ('乌黑', '稍蜷', '浊响', '清晰', '凹陷', '硬滑', 0.360, 0.370, '否'),
    ('浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '否'),
    ('青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '否')
]

# 测1数据
test_sample = ('青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460)

# 分离数据集
X_train = [d[:-1] for d in data]
y_train = [d[-1] for d in data]

# 统计类别的先验概率
class_counts = Counter(y_train)
total_samples = len(y_train)
prior_probs = {cls: count / total_samples for cls, count in class_counts.items()}


# 属性的可能取值
attr_values = [list(set([sample[i] for sample in X_train])) for i in range(6)]  # 只取前六个属性

# 计算类别下的均值和方差 (针对连续特征)
def mean(values):
    return sum(values) / len(values)

def variance(values, mean_value):
    return sum((x - mean_value) ** 2 for x in values) / len(values)

# 计算高斯分布概率密度函数
def gaussian_prob(x, mean, var):
    exponent = math.exp(-((x - mean) ** 2) / (2 * var))
    return (1 / math.sqrt(2 * math.pi * var)) * exponent


# 拉普拉斯修正的条件概率
def compute_laplace_prob(X_train, y_train, attr_idx, attr_value, target_class, attr_values):
    # 统计目标类别下属性值的出现次数
    count = sum(1 for i, sample in enumerate(X_train) if sample[attr_idx] == attr_value and y_train[i] == target_class)
    class_count = class_counts[target_class]
    # 拉普拉斯修正
    return (count + 1) / (class_count + len(attr_values[attr_idx]))

# 获取每个类别的连续属性统计量 (均值和方差)
continuous_attrs = defaultdict(lambda: defaultdict(list))  # 类别 -> 特征 -> 值
for sample in data:
    label = sample[-1]
    continuous_attrs[label]['density'].append(sample[6])
    continuous_attrs[label]['sugar'].append(sample[7])

# 计算每个类别下密度和含糖率的均值和方差
stats = defaultdict(dict)  # 类别 -> 特征 -> (均值, 方差)
for cls, features in continuous_attrs.items():
    for feature_name, values in features.items():
        mu = mean(values)
        var = variance(values, mu)
        stats[cls][feature_name] = (mu, var)

# 预测函数
def predict(test_sample):
    class_probs = {}
    for cls in class_counts:
        # 计算先验概率
        prob = prior_probs[cls]
        for i in range(6):  # 计算前六个离散属性
            prob *= compute_laplace_prob(X_train, y_train, i, test_sample[i], cls, attr_values)
        density_mean, density_var = stats[cls]['density']
        sugar_mean, sugar_var = stats[cls]['sugar']
        prob *= gaussian_prob(test_sample[6], density_mean, density_var)  # 密度条件概率
        prob *= gaussian_prob(test_sample[7], sugar_mean, sugar_var)  # 含糖率条件概率
        class_probs[cls] = prob
    # 返回概率最大的类别
    return max(class_probs, key=class_probs.get), class_probs

# 对"测1"进行分类
predicted_class, class_probabilities = predict(test_sample)
print(f"Predicted class for '测1': {predicted_class}")
print(f"Class probabilities: {class_probabilities}")
