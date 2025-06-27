from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Feature numeriche (includi anche quella del tuo transformer personalizzato)
numerical_features = [
    'LeadTime', 'StaysInWeekendNights', 'StaysInWeekNights', 'Adults',
    'Children', 'Babies', 'ADRThirdQuartileDeviation', 
    'PreviousCancellations', 'PreviousBookingsNotCanceled', 'BookingChanges',
    'DaysInWaitingList', 'TotalOfSpecialRequests'
]

# Feature categoriche da codificare
one_hot_features = [
    'Meal', 'MarketSegment', 'DistributionChannel',
    'DepositType', 'CustomerType', 'HotelType',
    'Agent', 'Company', 'IsRepeatedGuest'
]

# Preprocessor compatibile con imblearn.Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), one_hot_features)
    ]
)
