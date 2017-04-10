def family_size_feature(df):
	return df.apply(lambda x: x.parch + x.sibsp, axis=1)
	
def embarked_S_feature(df):
	return df.embarked.apply(lambda x: 1 if x=='S' else 0)
	
def embarked_C_feature(df):
	return df.embarked.apply(lambda x: 1 if x=='C' else 0)
	
def first_class_feature(df):
	return df.pclass.apply(lambda x: 1 if x==1 else 0)
	
def second_class_feature(df):
	return df.pclass.apply(lambda x: 1 if x==2 else 0)
	

	