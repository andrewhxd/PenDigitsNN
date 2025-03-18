import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

st.title('Neural Network Visualizer')

model = tf.keras.models.load_model('my_model.keras')
st.write("### Model Summary")
model.summary(print_fn=lambda x: st.text(x))

user_input = st.text_input("Enter input data (comma-separated):", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
if user_input:
    try:
        input_data = np.array([float(x) for x in user_input.split(',')]).reshape(1, -1)
        x_coords = input_data[0][::2]
        y_coords = input_data[0][1::2]
        
        prediction = model.predict(input_data)
        st.write("### Prediction:", np.argmax(prediction, axis=1)[0])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot connected points in order
        ax1.plot(x_coords, y_coords, marker='o', linestyle='-', color='blue')
        
        # Add point numbers
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            ax1.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
        
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')
        ax1.set_title('Input Points (Connected in Order)')
        ax1.grid(True)
        
        ax2.bar(range(len(prediction[0])), prediction[0])
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Probability')
        ax2.set_title('Prediction Probabilities')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error: {e}")