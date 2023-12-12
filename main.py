import pandas as pd
import re

training_set = pd.read_parquet('Tutorial_training_set.parquet')
print(training_set.shape)
testing_set = pd.read_parquet('Tutorial_testing_set.parquet')
# print(testing_set.shape)

# print(training_set.head())


# training_set['fa_dollar_sign'].to_csv('fa_dollar_sign', index=False)

## 簡介區

# 我們的目標
# converted_salary 

# 可以用的東西們
# profession, fa_sitemap, fa_user 直接 one-hot



## 超簡單 one-hot 區域

X = training_set[['profession', 'fa_sitemap', 'fa_user']]  # 或者其他需要的特徵列

X = pd.get_dummies(X, columns=['profession'])
X = pd.get_dummies(X, columns=['fa_sitemap'])
X = pd.get_dummies(X, columns=['fa_user'])


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立多元線性回歸模型
model = LinearRegression()
model.fit(X_train, y_train)

# 預測測試集
y_pred = model.predict(X_test)

# 計算 R²
r2 = r2_score(y_test, y_pred)
print(f'R²: {r2}')