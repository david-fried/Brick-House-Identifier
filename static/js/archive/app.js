var svgWidth = 1000;
var svgHeight = 700;

var margin = {
  top: 20,
  right: 40,
  bottom: 200,
  left: 150
};

var width = svgWidth - margin.right - margin.left;
var height = svgHeight - margin.top - margin.bottom;

var chart = d3
  .select('#scatter')
  .append('div')
  .classed('chart', true);

// Create an SVG wrapper, append an SVG group that will hold our chart,
// and shift the latter by left and top margins.
var svg = chart
    .append("svg")
    .attr("width", svgWidth)
    .attr("height", svgHeight);

// Append an SVG group
var chartGroup = svg.append("g")
  .attr("transform", `translate(${margin.left}, ${margin.top})`);

// Initial Params
var chosenXAxis = "fireball";
var chosenYAxis = "state_sightings";

// let zoom = d3.zoom()
//    .scaleExtent([1, 2])
//    .translateExtent([[-500, -300], [1500, 1000]])
//    .on('zoom', () => {
//        svg.attr('transform', d3.event.transform)
//    });
 
// svg.call(zoom);

// function used for updating x-scale var upon click on axis label
function xScale(ufoData, chosenXAxis) {
  // create scales
  var xLinearScale = d3.scaleLinear()
    .domain([d3.min(ufoData, d => d[chosenXAxis]) * 0.8,
      d3.max(ufoData, d => d[chosenXAxis]) * 1.0
    ])
    .range([0, width]);

  return xLinearScale;

}

// function used for updating x-scale var upon click on axis label
function yScale(ufoData, chosenYAxis) {
  // create scales
  var yLinearScale = d3.scaleLinear()
    .domain([d3.min(ufoData, d => d[chosenYAxis]) * 0.8,
      d3.max(ufoData, d => d[chosenYAxis]) * 1.0
    ])
    .range([height, 0]);

  return yLinearScale;

}

// function used for updating xAxis var upon click on axis label
function renderxAxes(newXScale, xAxis) {
  var bottomAxis = d3.axisBottom(newXScale);

  xAxis.transition()
    .duration(1000)
    .call(bottomAxis);

  return xAxis;
}

// function used for updating yAxis var upon click on axis label
function renderyAxes(newYScale, yAxis) {
  var leftAxis = d3.axisLeft(newYScale);

  yAxis.transition()
    .duration(1000)
    .call(leftAxis);

  return yAxis;
}

// function used for updating circles group with a transition to
// new circles
function renderCircles(circlesGroup, newXScale, chosenXAxis, newYScale, chosenYAxis) {

  circlesGroup.transition()
    .duration(1000)
    .attr("cx", d => newXScale(d[chosenXAxis]))
    .attr("cy", d => newYScale(d[chosenYAxis]));

  return circlesGroup;
}

//function for updating states
function renderText(textGroup, newXScale, chosenXAxis, newYScale, chosenYAxis) {

  textGroup.transition()
    .duration(1000)
    .attr('x', d => newXScale(d[chosenXAxis]))
    .attr('y', d => newYScale(d[chosenYAxis]));

  return textGroup
}
// function to retrieve data on tooltip values 
function style(value, chosenXAxis) {

  //style based on variable
  //poverty
  if (chosenXAxis === 'fireball') {
      return `${value}`;
  }
  //household income
  else if (chosenXAxis === 'triangle') {
      return `${value}`;
  }
  else {
    return `${value}`;
  }
}
// function used for updating circles group with new tooltip
function updateToolTip(chosenXAxis, chosenYAxis, circlesGroup) {

  //select x label
  //poverty percentage
  if (chosenXAxis === 'fireball') {
      var xLabel = "Fireball Sightings(#):";
  }
  //household income in dollars
  else if (chosenXAxis === 'triangle') {
      var xLabel = "Triangle Sightings(#):";
  }
  //age (number)
  else {
      var xLabel = "Light Formation Sightings(#)";
  }

  //select y label
  //percentage lacking healthcare
  if (chosenYAxis === 'state_sightings') {
      var yLabel = "State Sightings(#):"
  }
  //percentage obese
  else if (chosenYAxis === 'death_rate') {
      var yLabel = "Overdose Death Rate:";
  }
  //smoking percentage
  else {
      var yLabel = "Smoking (%):";
  }

  //create tooltip
  var toolTip = d3.tip()
      .attr("class", "d3-tip")
      .offset([-2, 0])
      .html(function(d) {
          return (`${d.state_abb}<br>${xLabel} ${style(d[chosenXAxis], chosenXAxis)}<br>${yLabel} ${d[chosenYAxis]}`);
      });

  circlesGroup.call(toolTip);

  //add events
  circlesGroup.on("mouseover", toolTip.show)
  .on("mouseout", toolTip.hide);

  return circlesGroup;
}

// Retrieve data from the SQL database and execute everything below
d3.json("http://127.0.0.1:5000/d3data").then(function(ufoData, err) {
  if (err) throw err;

  // parse data
  ufoData.forEach(function(data) {
    data.death_rate = +data.death_rate;
    data.population = +data.population;
    // divide population by 100,000
    data.state_sightings = +data.state_sightings;
    data.smokes = +data.smokes;
    // grab initial value for number of fireball sightings by state
    data.fireball_adj = +data.fireball; 
    // divide fireball sightings by number of residents per 100,000
    data.fireball = +data.fireball_adj * 100000 / data.population;
    data.light_adj = +data.light;
    data.light = +data.light_adj * 100000 / data.population;
    data.triangle_adj = +data.light;
    data.triangle = +data.triangle_adj * 100000 / data.population;
  });

  // xLinearScale function above csv import
  var xLinearScale = xScale(ufoData, chosenXAxis);
  var yLinearScale = yScale(ufoData, chosenYAxis);

  // Create initial axis functions
  var bottomAxis = d3.axisBottom(xLinearScale);
  var leftAxis = d3.axisLeft(yLinearScale);

  // append x axis
  var xAxis = chartGroup.append("g")
    .classed("x-axis", true)
    .attr("transform", `translate(0, ${height})`)
    .call(bottomAxis);

  //append y axis
  var yAxis = chartGroup.append("g")
      .classed("y-axis", true)
      // .scale(x)
	    // .ticks(4) //specify number of ticks 
      // .orient("bottom");
      .call(leftAxis);

  // append initial circles
  var circlesGroup = chartGroup.selectAll("circle")
    .data(ufoData)
    .enter()
    .append("circle")
    // .classed("stateCircle", true)
    .attr("cx", d => xLinearScale(d[chosenXAxis]))
    .attr("cy", d => yLinearScale(d[chosenYAxis]))
    .attr("r", 15.5)
    .attr("fill", "#44798e")
    .attr("opacity", ".75");

    //append text
    var textGroup = chartGroup.selectAll('.stateText')
      .data(ufoData)
      .enter()
      .append('text')
      .classed('stateText', true)
      .attr('x', d => xLinearScale(d[chosenXAxis]))
      .attr('y', d => yLinearScale(d[chosenYAxis]))
      .attr('dy', 3)
      .attr('font-size', '10px')
      .text(function(d){return d.state_abb});

  // Create group for three x-axis labels
  var xlabelsGroup = chartGroup.append("g")
  .attr('transform', `translate(${width / 2}, ${height + 20 + margin.top})`);

  var fireballLabel = xlabelsGroup.append("text")
    .classed("aText", true)
    .attr("x", 0)
    .attr("y", 20)
    .attr("value", "fireball") // value to grab for event listener
    .classed("active", true)
    .text("Number of Fireball Sightings per 100,000");

  var lightLabel = xlabelsGroup.append("text")
    .classed("aText", true)
    .attr("x", 0)
    .attr("y", 40)
    .attr("value", "light") // value to grab for event listener
    .classed("inactive", true)
    .text("Number of Light Formation Sightings per 100,000");

    var triangleLabel = xlabelsGroup.append("text")
    .classed("aText", true)
    .attr("x", 0)
    .attr("y", 60)
    .attr("value", "triangle") // value to grab for event listener
    .classed("inactive", true)
    .text("Number of Triangle Sightings per 100,000");


  // Create group for three y-axis labels
  var ylabelsGroup = chartGroup.append("g")
    .attr("transform", `translate(${0 - margin.left/4}, ${(height/2)})`);

  var statesightingsLabel = ylabelsGroup.append("text")
    .classed("aText", true)
    .classed("active", true)
    .attr("x", 0)
    .attr("y", 0 - 20)
    .attr("dy", "1em")
    .attr("transform", "rotate(-90)")
    .attr("value", "state_sightings")
    .text("Total Number of Sightings");

  var smokesLabel = ylabelsGroup.append("text")
    .classed("aText", true)
    .classed("inactive", true)
    .attr("x", 0)
    .attr("y", 0 - 40)
    .attr("dy", "1em")
    .attr("transform", "rotate(-90)")
    .attr("value", "smokes")
    .text("Smoking (%)");

  var censusLabel = ylabelsGroup.append("text")
    .classed("aText", true)
    .classed("inactive", true)
    .attr("x", 0)
    .attr("y", 0 - 60)
    .attr("dy", "1em")
    .attr("transform", "rotate(-90)")
    .attr("value", "death_rate")
    .text("Rate of Drug Overdose");

  // updateToolTip function above csv import
  var circlesGroup = updateToolTip(chosenXAxis, chosenYAxis, circlesGroup);

    //x axis labels event listener
    xlabelsGroup.selectAll("text")
        .on("click", function() {
            //get value of selection
            var value = d3.select(this).attr("value");

            //check if value is same as current axis
            if (value != chosenXAxis) {

                //replace chosenXAxis with value
                chosenXAxis = value;

                //update x scale for new data
                xLinearScale = xScale(ufoData, chosenXAxis);

                //update x axis with transition
                xAxis = renderxAxes(xLinearScale, xAxis);

                //update circles with new x values
                circlesGroup = renderCircles(circlesGroup, xLinearScale, chosenXAxis, yLinearScale, chosenYAxis);

                //update tooltips with new info
                circlesGroup = updateToolTip(chosenXAxis, chosenYAxis, circlesGroup);

                //update text 
                textGroup = renderText(textGroup, xLinearScale, chosenXAxis, yLinearScale, chosenYAxis);

                //change classes
                if (chosenXAxis === "fireball") {
                    fireballLabel.classed("active", true).classed("inactive", false);
                    lightLabel.classed("active", false).classed("inactive", true);
                    triangleLabel.classed("active", false).classed("inactive", true);
                }
                else if (chosenXAxis === "light") {
                    fireballLabel.classed("active", false).classed("inactive", true);
                    lightLabel.classed("active", true).classed("inactive", false);
                    triangleLabel.classed("active", false).classed("inactive", true);
                }
                else {
                    fireballLabel.classed("active", false).classed("inactive", true);
                    lightLabel.classed("active", false).classed("inactive", true);
                    triangleLabel.classed("active", true).classed("inactive", false);
                }
            }
        });

    //y axis labels event listener
    ylabelsGroup.selectAll("text")
    .on("click", function() {
        //get value of selection
        var value = d3.select(this).attr("value");

        //check if value is same as current axis
        if (value != chosenYAxis) {

            //replace chosenYAxis with value
            chosenYAxis = value;

            //update y scale for new data
            yLinearScale = yScale(ufoData, chosenYAxis);

            //update x axis with transition
            yAxis = renderyAxes(yLinearScale, yAxis);

            //update circles with new y values
            circlesGroup = renderCircles(circlesGroup, xLinearScale, chosenXAxis, yLinearScale, chosenYAxis);

            //update tooltips with new info
            circlesGroup = updateToolTip(chosenXAxis, chosenYAxis, circlesGroup);

            //update text 
            textGroup = renderText(textGroup, xLinearScale, chosenXAxis, yLinearScale, chosenYAxis);

            //change classes
            if (chosenYAxis === "death_rate") {
                censusLabel.classed("active", true).classed("inactive", false);
                smokesLabel.classed("active", false).classed("inactive", true);
                statesightingsLabel.classed("active", false).classed("inactive", true);
            }
            else if (chosenYAxis === "smokes") {
                censusLabel.classed("active", false).classed("inactive", true);
                smokesLabel.classed("active", true).classed("inactive", false);
                statesightingsLabel.classed("active", false).classed("inactive", true);
            }
            else {
                censusLabel.classed("active", false).classed("inactive", true);
                smokesLabel.classed("active", false).classed("inactive", true);
                statesightingsLabel.classed("active", true).classed("inactive", false);
            }
        }
    });
}).catch(function(error) {
  console.log(error);
});