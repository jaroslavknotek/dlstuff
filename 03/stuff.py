# Recreate the exact same model purely from the file
import tensorflow as tf

from uppercase_data import UppercaseData


model = tf.keras.models.load_model('path_to_my_model.h5')
alphabet_size=500
window=11
uppercase_data = UppercaseData(window, alphabet_size)


with open("uppercase_test.txt", "w", encoding="utf-8") as out_file:
    res = model.predict(uppercase_data.dev.data["windows"][:1000])
    print(list(sorted(res[:,1],reverse=True))[:10])



    # TODO: Generate correctly capitalized test set.
    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    # and write the result to `uppercase_test.txt` file.


    pass
