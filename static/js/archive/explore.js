
tbody = d3.select("tbody");

d3.json('http://127.0.0.1:5000/exploredata', function(ufoData){


  createTable(ufoData);

  var button = d3.select("#filter-btn");

  // Select the form
  var form = d3.select(".filters");

  // Create event handlers 
  button.on("click", runEnter);
  form.on("submit", runEnter);


  // Complete the event handler function for the form
  function runEnter() {

      // remove previous ufoData from the list to
      tbody.html("");

      // Prevent the page from refreshing
      d3.event.preventDefault();

      var inputState= d3.select("#state").property("value").toLowerCase();

      var inputCity= d3.select("#city").property("value").toLowerCase();

      var inputCountry= d3.select("#country").property("value").toLowerCase();

      var inputShape= d3.select("#shape").property("value").toLowerCase();

      filteredData = ufoData;

      if (inputState !== "") {
        var filteredData = filteredData.filter(obj=>obj[2] === inputState);
      }

      if (inputCity !== "") {
        var filteredData = filteredData.filter(obj=>obj[1] === inputCity);
      }
      
      if (inputCountry !== "") {
        var filteredData = filteredData.filter(obj=>obj[3] === inputCountry);
      }

      if (inputShape !== "") {
        var filteredData = filteredData.filter(obj=>obj[4] === inputShape);
      }
      
      createTable(filteredData);
  };


});


function createTable(ufoData) {
  ufoData.forEach((obj) => {
      var row = tbody.append("tr");
      Object.values(obj).forEach((value) => {
        var cell = row.append("td");
        cell.text(value);
      });
    });   
}