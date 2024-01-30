fetch('/getdata')
    .then(response => response.json())
    .then(data => {
        initializeApp(data);
    })
    .catch(error => {
        console.error('Error:', error);
    });

function initializeApp(data) {
    // Populate column dropdown
    populateColumnDropdown(data);

    // Set up date pickers with default values
    setupDatePickers(data);


    // Function to update charts based on user input
    window.updateCharts = function () {
        const selectedColumn = document.getElementById('columnSelect').value;
        const startDate = document.getElementById('startDate').value;
        const endDate = document.getElementById('endDate').value;

        // Filter and process data
        const filteredData = processData(data, selectedColumn, startDate, endDate);

        // Generate Histogram
        generateHistogram(filteredData, selectedColumn);

        // Generate Boxplot
        generateBoxplot(filteredData, selectedColumn);
    }
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

function setupDatePickers(data) {
    // Extracting dates and converting them to YYYY-MM-DD format
    const dates = data.map(item => {
        const date = new Date(item.Timestamp);
        return date.toISOString().split('T')[0]; // Format: YYYY-MM-DD
    });

    const minDate = new Date(Math.min(...dates.map(date => new Date(date))));
    const maxDate = new Date(Math.max(...dates.map(date => new Date(date))));

    // Formatting dates back to YYYY-MM-DD for the date picker
    const formattedMinDate = minDate.toISOString().split('T')[0];
    const formattedMaxDate = maxDate.toISOString().split('T')[0];

    // Finding date picker elements
    const startDatePicker = document.getElementById('startDate');
    const endDatePicker = document.getElementById('endDate');

    // Setting min and max values for date pickers
    startDatePicker.setAttribute('min', formattedMinDate);
    startDatePicker.setAttribute('max', formattedMaxDate);
    endDatePicker.setAttribute('min', formattedMinDate);
    endDatePicker.setAttribute('max', formattedMaxDate);

    // Setting default values
    startDatePicker.value = formattedMinDate;
    endDatePicker.value = formattedMaxDate;
}

function processData(data, selectedColumn, startDate, endDate) {
    // Convert start and end dates to Date objects for comparison
    const startDateObj = new Date(startDate);
    const endDateObj = new Date(endDate);

    // Filter data based on the selected date range and extract the selected column values
    return data
        .filter(item => {
            const itemDate = new Date(item.Timestamp);
            return itemDate >= startDateObj && itemDate <= endDateObj;
        })
        .map(item => item[selectedColumn]);
}


function generateHistogram(data, column) {
    // Assuming 'data' is an array of numeric values

    // Create bins for the histogram
    const bins = createHistogramBins(data);

    // Prepare the data for FusionCharts
    const dataSource = {
        chart: {
            caption: `Histogram of ${column}`,
            xAxisName: column,
            yAxisName: 'Count',
            theme: 'fusion'
        },
        data: bins
    };

    // Render the chart
    FusionCharts.ready(function () {
        const chart = new FusionCharts({
            type: 'column2d',
            renderAt: 'histogramChart',
            width: '700',
            height: '400',
            dataFormat: 'json',
            dataSource: dataSource
        });
        chart.render();
    });
}

function createHistogramBins(data) {
    // Define the number of bins or use a rule like Sturges' formula
    const numBins = Math.round(1 + 3.322 * Math.log10(data.length) / 2);

    // Find the range of the data
    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min;

    // Calculate the width of each bin
    const binWidth = range / numBins;

    // Initialize bins
    const bins = Array.from({length: numBins}, () => ({value: 0}));

    // Fill bins with data
    data.forEach(value => {
        const binIndex = Math.min(numBins - 1, Math.floor((value - min) / binWidth));
        bins[binIndex].value += 1;
    });

    // Format bins for FusionCharts
    return bins.map((bin, index) => ({
        label: `${(min + binWidth * index).toFixed(2)} - ${(min + binWidth * (index + 1)).toFixed(2)}`,
        value: bin.value
    }));
}


function generateBoxplot(data, column) {
    // Prepare the data for FusionCharts
    const dataSource = {
        chart: {
            caption: `Boxplot of ${column}`,
            xAxisName: column,
            yAxisName: 'Value',
            theme: 'fusion'
        },
        categories: [{category: [{label: column}]}],
        dataset: [{data: [{value: data.join(",")}]}]
    };

    // Render the chart
    FusionCharts.ready(function () {
        const chart = new FusionCharts({
            type: 'boxandwhisker2d',
            renderAt: 'boxplotChart',
            width: '700',
            height: '400',
            dataFormat: 'json',
            dataSource: dataSource
        });
        chart.render();
    });
}
