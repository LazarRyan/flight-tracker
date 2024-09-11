// Global variables
let chart = null;

// Function to fetch prediction from the backend
function fetchPrediction(data) {
    fetch('http://localhost:5001/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log('Prediction data:', data);
        updateUI(data);
    })
    .catch(error => {
        console.error('Error:', error);
        displayError('An error occurred while fetching the prediction. Please try again.');
    });
}

// Function to update the UI with prediction results
function updateUI(data) {
    const loadingElement = document.getElementById('loading');
    const resultsElement = document.getElementById('results');
    
    if (loadingElement) loadingElement.style.display = 'none';
    if (resultsElement) resultsElement.style.display = 'block';

    updateElement('average-price', `$${data.average.toFixed(2)}`);
    updateElement('min-price', `$${data.min.toFixed(2)}`);
    updateElement('max-price', `$${data.max.toFixed(2)}`);
    updateElement('accuracy', `±$${data.accuracy.toFixed(2)}`);

    createOrUpdateChart(data.dates, data.prices);
    displayBestDays(data.dates, data.prices);
}

// Helper function to safely update element text content
function updateElement(id, text) {
    const element = document.getElementById(id);
    if (element) element.textContent = text;
}

// Function to create or update the chart
function createOrUpdateChart(dates, prices) {
    let ctx = document.getElementById('price-chart');
    if (!ctx) {
        console.log('Chart canvas not found, creating one');
        const resultsElement = document.getElementById('results');
        if (resultsElement) {
            ctx = document.createElement('canvas');
            ctx.id = 'price-chart';
            resultsElement.appendChild(ctx);
        } else {
            console.error('Results element not found, cannot create chart');
            return;
        }
    }

    if (chart) {
        chart.destroy();
    }

    chart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Predicted Price',
                data: prices,
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Price ($)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                }
            }
        }
    });
}

// Function to display best days to book
function displayBestDays(dates, prices) {
    let bestDaysList = document.getElementById('best-days-list');
    if (!bestDaysList) {
        console.log('Best days list element not found, creating one');
        const resultsElement = document.getElementById('results');
        if (resultsElement) {
            bestDaysList = document.createElement('ul');
            bestDaysList.id = 'best-days-list';
            resultsElement.appendChild(bestDaysList);
        } else {
            console.error('Results element not found, cannot create best days list');
            return;
        }
    }

    const bestDays = dates.map((date, index) => ({ date, price: prices[index] }))
        .sort((a, b) => a.price - b.price)
        .slice(0, 5);

    bestDaysList.innerHTML = '';

    bestDays.forEach(day => {
        const li = document.createElement('li');
        li.textContent = `${day.date}: $${day.price.toFixed(2)}`;
        bestDaysList.appendChild(li);
    });
}

// Function to display error messages
function displayError(message) {
    const errorElement = document.getElementById('error-message');
    const loadingElement = document.getElementById('loading');
    
    if (errorElement) {
        errorElement.textContent = message;
        errorElement.style.display = 'block';
    }
    
    if (loadingElement) {
        loadingElement.style.display = 'none';
    }
}

// Event listener for form submission
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();

            const errorElement = document.getElementById('error-message');
            const loadingElement = document.getElementById('loading');
            const resultsElement = document.getElementById('results');

            if (errorElement) errorElement.style.display = 'none';
            if (loadingElement) loadingElement.style.display = 'block';
            if (resultsElement) resultsElement.style.display = 'none';

            const origin = document.getElementById('origin')?.value.toUpperCase();
            const destination = document.getElementById('destination')?.value.toUpperCase();
            const date = document.getElementById('date')?.value;

            if (!origin || !destination || !date) {
                displayError('Please fill in all fields.');
                return;
            }

            fetchPrediction({origin, destination, date});
        });
    }

    // Initialize date picker with a minimum date of today
    const dateInput = document.getElementById('date');
    if (dateInput) {
        const today = new Date().toISOString().split('T')[0];
        dateInput.min = today;
    }
});
