// from data.js
var tableData = data;

// Use D3 to select the table body
var tbody = d3.select("tbody");

// Input data to HTML
data.forEach(function(i) {
  console.log(i);
  var row = tbody.append("tr");

  Object.entries(i).forEach(function([key, value]) {
      console.log(key, value);
      var cell = row.append("td");
      cell.text(value);
  });
});

// Select button
var button = d3.select("#filter-btn");
button.on("click", pressFilter);

function pressFilter() {

    // Prevent the page from refreshing
    d3.event.preventDefault();

    // Table body
    tbody.html("");

    // Select the input element and get the raw HTML node
    var inputElement = d3.select("#datetime");
  
    // Get the value property of the input element
    var inputValue = inputElement.property("value");
  
    console.log(inputValue);
  
    var filteredData = tableData.filter(j => j.datetime === inputValue);
  
    console.log(filteredData);

    // Input filtered data to HTML
    filteredData.forEach(function(m) {
        console.log(m);
        var row = tbody.append("tr");
      
        Object.entries(m).forEach(function([key, value]) {
            console.log(key, value);
            var cell = row.append("td");
            cell.text(value);
        });
      });


  };
