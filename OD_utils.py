# general purpose utils

class rounded_dict(dict):	
	def __str__(self, round_value=3):
		return str({k: round(v, round_value) if isinstance(v, float) else v for k,v in self.items()})