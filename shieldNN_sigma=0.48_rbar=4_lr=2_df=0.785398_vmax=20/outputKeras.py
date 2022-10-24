import tf1
import tf2
import tf3

if __name__ == '__main__':
    model1 = tf1.load_model()
    model1.compile(optimizer='adam',loss='mean_squared_error')
    model1.save('1.h5',save_format='h5')

    model2 = tf2.load_model()
    model2.compile(optimizer='adam',loss='mean_squared_error')
    model2.save('2.h5',save_format='h5')

    model3 = tf3.load_model()
    model3.compile(optimizer='adam',loss='mean_squared_error')
    model3.save('3.h5',save_format='h5')


