# Neural Networks as Cryptographic Hash Functions
* ```Hash.py``` contains the different model classes that we used. It makes use of ```msg_to_bits.py``` to preprocess the input
if it's passed as a string.
* As the filenames suggest, the rest of the scripts deal with testing different hash function properties that we set out
to emulate.
	* They can all be run with the command `python <test_script_here.py>`. 
	* `collision_check.py` will run the Dense model by default. Parameters `lstm` and `double` can be passed to run the test on the other respective models.
		* The command to run the collision check on the DoubleDense model would be `python collision_check.py double`
		
* Here is a sample script to simply test the different models and hash arbitrary strings:

```
from Hash import DenseHash
from Hash import LSTMHash
from Hash import DoubleDenseHash

message = "test message to be hashed goes here"

model = DenseHash()  # change this to any of the models imported from Hash.py

print(model.hash(message))  # will return the bit sequence of the computed hash
```