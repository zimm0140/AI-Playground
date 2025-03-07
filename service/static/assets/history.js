/**
 * AI Playground History Renderer
 * ------------------------------
 * This script populates the history view with previously generated images and their parameters.
 * 
 * It processes the 'history' array (defined in main.js) and creates HTML elements for each history item,
 * showing the output image and all parameter values used to generate it. Image parameters are rendered
 * as clickable links to the original images.
 * 
 * The rendered history provides users with a visual record of their generations and the exact
 * parameters used, allowing them to replicate or modify previous results.
 */

document.addEventListener("DOMContentLoaded", function () {
  // Initialize HTML container string
  html = "";
  
  // Loop through each item in the history array
  for (let i = 0; i < history.length; i++) {
    let item = history[i];
    
    // Start a new history item with the output image
    html += `<li><div class="result-img"><img src="${item.out_image}" /></div><ul class="params">`;
    
    // Loop through each parameter for this history item
    for (let j = 0; j < item.params.length; j++) {
      let param = item.params[j];
      
      // Handle image parameters differently - extract filename and make them clickable
      if (param.type == "image") {
        pos = param.value.lastIndexOf("/");
        filename = pos > -1 ? param.value.substring(pos + 1) : param.value;
        html += `<li><span class="param-name">${param.name}</span><span class="param-value"><a href='${param.value}' target='_blank'>${filename}</a></span></li>`;
      } else {
        // Regular parameter display as name-value pair
        html += `<li><span class="param-name">${param.name}</span><span class="param-value">${param.value}</span></li>`;
      }
    }
    
    // Close the parameter list and history item
    html += "</ul></li>";
  }
  
  // Insert the generated HTML into the history list container
  document.querySelector(".history-list").innerHTML = html;
});
