import pandas as pd
import re

training_set = pd.read_parquet('Tutorial_training_set.parquet')
print(training_set.shape)
testing_set = pd.read_parquet('Tutorial_testing_set.parquet')
# print(testing_set.shape)

# print(training_set.head())


# training_set['fa_dollar_sign'].to_csv('fa_dollar_sign', index=False)
training_set['fa_business_time'].value_counts().to_csv('(raw)fa_business_time.csv')


## 簡介區

# 我們的目標
# converted_salary 

# 可以用的東西們
# profession, fa_sitemap, fa_user 直接 one-hot





## 轉換職位高低

# 用正則表達式處理白痴資料
pattern = r"全職・(.+?)$"
training_set['fa_user'] = training_set['fa_user'].apply(lambda x: re.search(pattern, x).group(1) if re.search(pattern, x) else x)

# 把 job_tags 轉換為長度
training_set['len_job_tags'] = training_set['job_tags'].apply(len)

# 對職位高低進行 Label Encoding
position_mapping = {
    '實習': 1,
    '初階': 2,
    '助理': 3,
    '中高階': 4,
    '經理 / 總監': 5,
    '經營層 (VP, GM, C-Level)': 6
}

sitemap_mapping = {
    '不需負擔管理責任': 1,
    '管理 1 ~ 5 人': 2,
    '管理 5 ~ 10 人': 3,
    '管理 10 ~ 15 人': 4,
    '管理 15 人以上': 5,
    '管理人數未定': 0
}

profession_mapping = {
    'it': 'Technology and Engineering',
    'engineering': 'Technology and Engineering',
    'bio-medical': 'Technology and Engineering',
    'manufacturing': 'Technology and Engineering',
    'marketing-advertising': 'Business and Management',
    'management-business': 'Business and Management',
    'finance': 'Business and Management',
    'hr': 'Business and Management',
    'design': 'Creative and Design',
    'media-communication': 'Creative and Design',
    'customer-service': 'Customer Service and Sales',
    'sales': 'Customer Service and Sales',
    'education': 'Education and Public Service',
    'public-social-work': 'Education and Public Service',
    'food-beverage': 'Other',
    'game-production': 'Other',
    'construction': 'Other',
    'law': 'Other',
    'other': 'Other'
}

business_time_mapping = {
    '不限年資': 0,
    '需具備 1 年以上工作經驗': 1,
    '需具備 2 年以上工作經驗': 2,
    '需具備 3 年以上工作經驗': 3,
    '需具備 4 年以上工作經驗': 4,
    '需具備 5 年以上工作經驗': 5,
    '需具備 6 年以上工作經驗': 6,
    '需具備 7 年以上工作經驗': 7,
    '需具備 8 年以上工作經驗': 8,
    '需具備 10 年以上工作經驗': 10,
    '需具備 12 年以上工作經驗': 12,
    '需具備 15 年以上工作經驗': 15,
    '需具備 20 年以上工作經驗': 20,
    '需具備 22 年以上工作經驗': 22
}


training_set['fa_user_encoded'] = training_set['fa_user'].map(position_mapping)
training_set['fa_sitemap_encoded'] = training_set['fa_sitemap'].map(sitemap_mapping).fillna(0)
training_set['profession_sort'] = training_set['profession'].map(profession_mapping).fillna('Other')
training_set['fa_business_time_encoded'] = training_set['fa_business_time'].map(business_time_mapping).fillna(0)
#training_set['profession_sort'].value_counts().to_csv('whathapopen')
## 超簡單 one-hot 區域
#X = training_set[['profession', 'fa_sitemap', 'fa_user']]  # 或者其他需要的特徵列

#training_set['job_tags'].to_csv('(raw)job_tags.csv')

X = training_set[['fa_sitemap_encoded', 'fa_user_encoded','profession_sort',"fa_business_time_encoded"]]  # 或者其他需要的特徵列
# "len_job_tags"
X = pd.get_dummies(X, columns=['profession_sort'])
#X = pd.get_dummies(X, columns=['fa_sitemap'])
#X = pd.get_dummies(X, columns=['fa_user'])



## 轉換薪資
# 假設 training_set 是您的 DataFrame，且包含 'fa_dollar_sign' 這個 Series
# 例如: training_set = pd.DataFrame({'fa_dollar_sign': ['79萬+ TWD / 月', '3萬 ~ 4萬 TWD / 月', '80萬 ~ 120萬 TWD / 年']})


def convert_salary(row):
    # 解析數字和單位
    numbers = re.findall(r'\d+', row)
    is_monthly = '月' in row
    is_plus = '萬+' in row
    is_range = '~' in row

    # 轉換為 float
    numbers = [float(num) for num in numbers]

    # 處理特殊情況
    if is_plus:
        salary = numbers[0]
    elif is_range:
        salary = sum(numbers) / len(numbers)
    # else:
    #     salary = numbers[0]
    #     print("aaaaa")

    # 月薪轉換為年薪
    if is_monthly:
        salary *= 14

    return salary

# 對 'fa_dollar_sign' 應用轉換函數
training_set['converted_salary'] = training_set['fa_dollar_sign'].apply(convert_salary)

print(training_set.shape)

# 開 train 了同學們
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 假設 training_set 是您的 DataFrame
# training_set = ...

# 選擇特徵和標籤
y = training_set['converted_salary']

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 建立多元線性回歸模型
model = LinearRegression()
model.fit(X_train, y_train)

# 預測測試集
y_pred = model.predict(X_test)

# 計算 R²
r2 = r2_score(y_test, y_pred)
print(f'R²: {r2}')

# print(X.columns.tolist)