from sklearn.model_selection import train_test_split

def tts(df, ser, random_state=0):
    return train_test_split(df, ser, random_state=random_state)
