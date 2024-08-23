pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Test') {
            steps {
                sh 'python -m unittest discover'
            }
        }

        stage('Deploy') {
            steps {
                sh 'python train_model.py'
            }
        }

        stage('Deploy to Flask') {
            steps {
                sh 'python app.py'
            }
        }
    }
}
