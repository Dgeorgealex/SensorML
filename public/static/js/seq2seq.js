fetch('/getdata')
    .then(response => response.json())
    .then(data => {
        initializeApp(data, '/seq2seq_predict_route');
    })
    .catch(error => {
        console.error('Error:', error);
    });

function initializeApp(data, route) {
    // Populate column dropdown
    populateColumnDropdown(data);

    // Set up date pickers with default values
    setupDatePickers(data);

    // Function to update charts based on user input
    window.updateCharts = function () {
        const selectedColumn = document.getElementById('columnSelect').value;
        const startDate = document.getElementById('startDate').value;

        fetch(route, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({startDate: startDate}),
        })
            .then(response => response.json())
            .then(data => {
                console.log(data);
                const predictions = data[0]
                const actual = data[1]
                render_chard(predictions, actual, selectedColumn);
            })
            .catch((error) => {
                console.error('Error:', error);
            });
    }
}

function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    const day = date.getDate().toString().padStart(2, '0');
    const month = (date.getMonth() + 1).toString().padStart(2, '0'); // +1 because months are 0-indexed
    return `${hours}:${minutes} ${day}/${month}`;
}

function render_chard(predictions, actual, featureName) {
    const predictionData = predictions.map(item => ({
        label: formatTimestamp(item.Timestamp),
        value: item[featureName]
    }));

    const actualData = actual.map(item => ({
        label: formatTimestamp(item.Timestamp),
        value: item[featureName]
    }));
    /*const predictionData = predictions.map(item => {
        console.log(item[featureName])
        return {label: item.Timestamp, value: item[featureName]};
    });
    const actualData = actual.map(item => {
        return {label: item.Timestamp, value: item[featureName]};
    });*/

    const dataSource = {
        chart: {
            caption: "Predictions vs Actual",
            subCaption: featureName,
            xAxisName: "Timestamp",
            yAxisName: "Value",
            theme: "fusion",
        },
        categories: [
            {
                category: predictionData.map(item => ({label: item.label}))
            }
        ],
        dataset: [
            {
                seriesname: "Predictions",
                data: predictionData
            },
            {
                seriesname: "Actual",
                data: actualData
            }
        ]
    };
    const chart = new FusionCharts({
        type: 'msline', // Multi-series line chart
        renderAt: 'chart-container', // ID of the container where the chart needs to be rendered
        width: '700', // Width of the chart
        height: '400', // Height of the chart
        dataFormat: 'json',
        dataSource: dataSource
    });
    chart.render();
}


function setupDatePickers(data) {
    // Extracting dates and converting them to YYYY-MM-DD format
    const dates = data.map(item => {
        const date = new Date(item.Timestamp);
        return date.toISOString().split('T')[0]; // Format: YYYY-MM-DD
    });

    const minDate = new Date(Math.min(...dates.map(date => new Date(date))));
    const maxDate = new Date(Math.max(...dates.map(date => new Date(date))));
    maxDate.setDate(maxDate.getDate() - 10);

    // Formatting dates back to YYYY-MM-DD for the date picker
    const formattedMinDate = minDate.toISOString().split('T')[0];
    const formattedMaxDate = maxDate.toISOString().split('T')[0];

    // Finding date picker elements
    const startDatePicker = document.getElementById('startDate');

    // Setting min and max values for date pickers
    startDatePicker.setAttribute('min', formattedMinDate);
    startDatePicker.setAttribute('max', formattedMaxDate);

    // Setting default values
    startDatePicker.value = formattedMinDate;
}

function populateColumnDropdown(data) {
    // Assuming first object has all columns
    const columns = Object.keys(data[0]);

    // Finding dropdown element
    const dropdown = document.getElementById('columnSelect');

    // Adding non-timestamp columns as options
    columns.forEach(column => {
        if (column !== 'Timestamp') {
            const option = document.createElement('option');
            option.value = column;
            option.textContent = column;
            dropdown.appendChild(option);
        }
    });
}