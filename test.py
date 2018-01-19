import tensorflow as tf
import os

data_dir = "/home/anthony/deeplentedtree/log/Grid_Search/clique-N1000-K30-E0-M1-exTrue-L_False-F_False"
searches =  os.listdir(data_dir)

best_score = 0
best = None
best_step = 0

for root, dirs, files in os.walk(data_dir): 
	for file in files:
		print file
		path_to_event = os.path.join(root,file)
		if 'test' in path_to_event:		
			for e in tf.train.summary_iterator(path_to_event):
				for i, v in enumerate(e.summary.value):
					if v.tag == 'accuracy/accuracy':
						if v.simple_value > best_score:
							best_score = v.simple_value
							best = path_to_event
							best_step = i

print best_score
print best
print best_step

for e in tf.train.summary_iterator(best):
	t = e.summary.value[best_step]
	print t
