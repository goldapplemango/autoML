from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# 결측값 처리 함수 (다양한 전략을 선택할 수 있도록 개선)
def handle_missing_values(data, strategy='mean'):
    """
    결측값 처리
    :param data: 데이터프레임
    :param strategy: 'mean', 'median', 'mode', 'drop' 중 하나 (기본값은 'mean')
    :return: 결측값이 처리된 데이터프레임
    """
    if strategy == 'drop':
        return data.dropna()
    else:
        imputer = SimpleImputer(strategy=strategy)
        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# 범주형 변수 처리 함수
def handle_categorical_data(data, encoding_strategy='label'):
    """
    범주형 데이터 처리
    :param data: 데이터프레임
    :param encoding_strategy: 'label' or 'onehot' (기본값은 'label')
    :return: 인코딩된 데이터프레임
    """
    if encoding_strategy == 'label':
        label_encoder = LabelEncoder()
        for column in data.select_dtypes(include=['object']).columns:
            data[column] = label_encoder.fit_transform(data[column])
    elif encoding_strategy == 'onehot':
        data = pd.get_dummies(data, drop_first=True)  # 첫 번째 컬럼은 드롭하여 더미변수 수를 줄임
    return data

# 표준화 함수
def scale_features(features, scaling_strategy='standard'):
    """
    데이터 스케일링 (표준화 또는 정규화)
    :param features: 피처 데이터
    :param scaling_strategy: 'standard', 'minmax', 'robust' 중 하나 (기본값은 'standard')
    :return: 스케일링된 피처 데이터
    """
    if scaling_strategy == 'standard':
        scaler = StandardScaler()
    elif scaling_strategy == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_strategy == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid scaling strategy provided. Use 'standard', 'minmax', or 'robust'.")
    
    return scaler.fit_transform(features)

# 데이터 전처리 함수 (종합)
def preprocess_data(file_path, scaling_strategy='standard', encoding_strategy='label', missing_strategy='mean'):
    try:
        # 데이터 로드
        data = pd.read_csv(file_path)
        
        # 결측값 처리
        data = handle_missing_values(data, strategy=missing_strategy)
        
        # 범주형 변수 처리
        data = handle_categorical_data(data, encoding_strategy=encoding_strategy)
        
        # 피처와 타겟 분리
        features = data.drop('target', axis=1)  # 'target' 컬럼을 제외한 피처들
        target = data['target']  # 타겟 변수
        
        # 스케일링
        features_scaled = scale_features(features, scaling_strategy=scaling_strategy)
        
        return features_scaled, target
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        return None, None

