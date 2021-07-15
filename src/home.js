const electron = require("electron");
const $ = require("jquery");
var fs = require("fs");
const GMM_folder = __dirname + "/GMM_files/";
let trained_audio = [];

fs.readdir(GMM_folder, (err, files) => {
  files.forEach((file) => {
    trained_audio.push(file.slice(0, -4));
    console.log(file.slice(0, -4));
    let newName = `
            <li >
                ${file.slice(0, -4)}
            </li>
            `;

    $("#name-list").append(newName);
  });
});
