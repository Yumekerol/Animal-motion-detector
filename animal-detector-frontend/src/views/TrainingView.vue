<template>
  <div class="training-view">
    <h2>Model Training</h2>

    <div class="model-options">
      <div
        v-for="option in modelOptions"
        :key="option.id"
        class="model-option"
        :class="{ selected: selectedOption === option.id }"
        @click="selectOption(option.id)"
      >
        <h3>{{ option.name }}</h3>
        <p>{{ option.description }}</p>
        <p>Epochs: {{ option.epochs }}</p>
      </div>
    </div>

    <button
      @click="startTraining"
      class="train-button"
      :disabled="!selectedOption || training"
    >
      {{ training ? 'Training in progress...' : 'Start Training' }}
    </button>

    <div v-if="message" class="message" :class="{ success: isSuccess, error: !isSuccess }">
      {{ message }}
    </div>

    <div v-if="trainingProgress" class="progress-container">
      <h3>Training Progress</h3>
      <div class="progress-bar">
        <div
          class="progress"
          :style="{ width: trainingProgress.percentage + '%' }"
        ></div>
      </div>
      <p>{{ trainingProgress.percentage }}% complete</p>
      <p>Epoch: {{ trainingProgress.current_epoch }}/{{ trainingProgress.total_epochs }}</p>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import axios from 'axios'

export default {
  name: 'TrainingView',
  setup() {
    const modelOptions = [
      { id: 1, name: 'Nano', description: 'Fast training, lower accuracy', epochs: 100, size: 'n' },
      { id: 2, name: 'Small', description: 'Balanced speed and accuracy', epochs: 200, size: 's' },
      { id: 3, name: 'Medium', description: 'More accurate, slower', epochs: 300, size: 'm' },
      { id: 4, name: 'Large', description: 'Very accurate, slowest', epochs: 300, size: 'l' }
    ]

    const selectedOption = ref(null)
    const training = ref(false)
    const message = ref('')
    const isSuccess = ref(false)
    const trainingProgress = ref(null)

    const selectOption = (id) => {
      selectedOption.value = id
    }

    const startTraining = async () => {
      if (!selectedOption.value) return

      const option = modelOptions.find(o => o.id === selectedOption.value)
      training.value = true
      message.value = ''

      try {
        const response = await axios.post('http://localhost:5000/api/train', {
          epochs: option.epochs,
          model_size: option.size
        })

        isSuccess.value = response.data.success
        message.value = response.data.message

        // Simulate progress updates (in a real app, you'd use WebSockets)
        if (response.data.success) {
          trainingProgress.value = {
            current_epoch: 0,
            total_epochs: option.epochs,
            percentage: 0
          }

          const interval = setInterval(() => {
            if (trainingProgress.value.percentage >= 100) {
              clearInterval(interval)
              training.value = false
              return
            }

            trainingProgress.value = {
              current_epoch: Math.min(
                option.epochs,
                Math.floor(trainingProgress.value.current_epoch + (option.epochs / 50))
              ),
              total_epochs: option.epochs,
              percentage: Math.min(
                100,
                trainingProgress.value.percentage + 2
              )
            }
          }, 500)
        }
      } catch (error) {
        isSuccess.value = false
        message.value = 'Error during training: ' + error.message
        training.value = false
      }
    }

    return {
      modelOptions,
      selectedOption,
      training,
      message,
      isSuccess,
      trainingProgress,
      selectOption,
      startTraining
    }
  }
}
</script>

<style scoped>
.training-view {
  max-width: 1000px;
  margin: 0 auto;
  padding: 2rem;
}

h2 {
  margin-bottom: 2rem;
  color: #2c3e50;
}

.model-options {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.model-option {
  background: white;
  padding: 1.5rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  cursor: pointer;
  transition: all 0.2s;
}

.model-option:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.model-option.selected {
  border: 2px solid #3498db;
  background: #f0f8ff;
}

.model-option h3 {
  color: #2c3e50;
  margin-bottom: 0.5rem;
}

.model-option p {
  color: #7f8c8d;
  margin-bottom: 0.5rem;
  font-size: 0.9rem;
}

.train-button {
  padding: 0.75rem 1.5rem;
  background: #27ae60;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background 0.2s;
  font-size: 1rem;
  margin-bottom: 2rem;
}

.train-button:hover:not(:disabled) {
  background: #219653;
}

.train-button:disabled {
  background: #95a5a6;
  cursor: not-allowed;
}

.message {
  padding: 1rem;
  border-radius: 4px;
  margin-bottom: 1rem;
}

.message.success {
  background: #d4edda;
  color: #155724;
}

.message.error {
  background: #f8d7da;
  color: #721c24;
}

.progress-container {
  background: white;
  padding: 1.5rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  margin-top: 2rem;
}

.progress-container h3 {
  margin-bottom: 1rem;
  color: #2c3e50;
}

.progress-bar {
  height: 20px;
  background: #ecf0f1;
  border-radius: 10px;
  margin-bottom: 1rem;
  overflow: hidden;
}

.progress {
  height: 100%;
  background: #3498db;
  border-radius: 10px;
  transition: width 0.3s;
}

.progress-container p {
  margin-bottom: 0.5rem;
  color: #7f8c8d;
}
</style>