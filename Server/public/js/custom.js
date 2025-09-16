let groups = {};

let alertCounter = 0;

(async function ($) {
  "use strict"; // Start of use strict

  $("#file").on("change", function () {
    const fileName = $(this).val().split(/([\\/])/g).pop().replace(".ino.d32.bin", "")
    const nameInput = $("#versionName")
    if (!nameInput.val() || nameInput.val().length === 0) {
      nameInput.val(fileName)
    }
  })

  $("#startUpload").on("click", (e) => {
    const data = new FormData();
    const version = $("#versionName").val()

    data.append("file", document.getElementById("file").files[0], version);


    fetch('/api/upload', {
      method: 'POST',
      body: data
    }).then(res => res.json())
      .then(data => {
        // Clear fields
        document.getElementById("file").value = "";
        $("#versionName").val("")

        // Show success message
        showSuccess("uploadMessageBox", `sketch added with name: ${data.fileName}`)
      })

    // Stop submit action
    e.preventDefault();
  })


  $("#set-version").on("click", (e) => {

    const groupID = $("#groupSelect").val();
    const version = $("#fileSelect").val()
    const update = $("#autoUpload:checked").val()

    if (groupID.length === 0) {
      showError("versionMessageBox", "No group was selected")
      return;
    }


    fetch('/api/set-group-version', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        group: groupID,
        version: version,
        autoUpdate: update
      })
    }).then(res => res.json())
      .then(data => {
        if (data.message) {
          showSuccess("versionMessageBox", data.message)

          // Set the groups that are being updated inactive
          for (const groupID of data.groupIds) {
            groupUpdating(groupID);
          }
        }
      })

    // Stop submit action
    e.preventDefault();
  })

  const webSocket = new WebSocket("ws://localhost:8080")

  webSocket.onmessage = (message) => {

    const temp = String(message.data)
    // We got boards
    if (temp.substr(0, 2) === "B|") {
      const data = JSON.parse(temp.substr(2))
      const clients = data.Boards
      const groups = data.Groups
      updateViews(clients, undefined, undefined, groups)
    }

    //We got Alerts
    if (temp.substr(0, 2) === "A|") {
      const alerts = JSON.parse(temp.substr(2))
      updateViews(undefined, undefined, alerts, undefined)
    }

    //We got Files
    if (temp.substr(0, 2) === "F|") {
      const files = JSON.parse(temp.substr(2))
      updateViews(undefined, files, undefined, undefined)
    }
  }
})(jQuery); // End of use strict


function showSuccess(divId, message) {
  const container = $(`#${divId}`)
  container.addClass("success")
  container.text(message)
  container.fadeIn();
  setTimeout(() => {
    container.fadeOut()
  }, 5000)
}

function showError(divId, message) {
  const container = $(`#${divId}`)
  container.addClass("box-error")
  container.text(message)
  container.fadeIn();
  setTimeout(() => {
    container.fadeOut()
  }, 5000)
}

/**
 * Updates the selection view for picking groups to upload to.
 * @param groups - List of clients that is present.
 */
function updateGroups(groups) {
  const groupSelect = $("#groupSelect")
  groupSelect.html("")
  for (let group of Object.values(groups)) {
    groupSelect.append(`<option value="${group.id}">${group.name}</option>`)
  }
}

/**
 * Updates the client list view.
 * @param clients - Object of clients that is present.
 * @param files - The list of previously uploaded files.
 * @param alerts - The alerts generated on the server.
 * @param groups - Object containing all the groups
 */
function updateViews(clients, files, alerts, groups) {

  // Update the Client views
  if (typeof clients !== "undefined") {
    const clientDiv = $("#clientList");

    $("#connectedDevices").html(Object.values(clients).length)

    clientDiv.html("");


    for (let [key, value] of Object.entries(clients)) {
      const container = $(`<div id="board-${key}" class="row"></div>`)
      const leftColumn = $('<div class="col-9"></div>')
      clientDiv.append(container)
      container.append(leftColumn)


      const nameField = $(`<h4>${value.name}</h4>`)
      leftColumn.append(nameField)
      leftColumn.append(`<span class="tooltiptext">${key}</span>`)
      const groupBox = $(`
                <div>
                    <b>Group:</b> ${value.group} | <b>Version:</b> ${value.version}
                </div>
            `)
      leftColumn.append(groupBox)

      const rightColumn = $('<div class="col-3"></div>')
      container.append(rightColumn)

      // Buttons
      const rename = $(`<button type="button" class="btn btn-primary" style="font-size: smaller; margin-top: 5px ;margin-bottom: 10px">Rename Board</button>`)
      const reGroup = $(`<button type="button" class="btn btn-primary" style="font-size: smaller">Change Group</button>`)

      //set background on warning when board is offline
      if (!value.online) {
        container.css("background-color", "#cb3737")
        reGroup.prop("disabled", "true")
        rename.prop("disabled", "true")
        container.css("color", "black")
      }

      // Set name handler
      rename.on('click', (e) => {
        const name = prompt("Provide board Name", "Jhon")
        if (!name)
          return;

        fetch("/api/name", {
          method: "POST",
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            name: name,
            client: key
          })
        })

        e.preventDefault()
      })

      reGroup.on('click', (e) => {
        const group = prompt("Provide group id between 1 and 255", "1")
        if (!group)
          return

        fetch("/api/group", {
          method: "POST",
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            group: group,
            client: key
          })
        })

        e.preventDefault()
      })

      rightColumn.append(rename)
      rightColumn.append(reGroup)

      clientDiv.append("<hr>")

    }
  }

  // Update the Client views
  if (typeof groups !== "undefined") {
    const groupsDiv = $("#groupList");

    groupsDiv.html("");


    for (let [key, value] of Object.entries(groups)) {
      const container = $(`<div id="group-${key}" class="row"></div>`)
      const leftColumn = $('<div class="col-9"></div>')
      groupsDiv.append(container)
      container.append(leftColumn)
      const nameField = $(`<h4>${value.name}</h4>`)
      leftColumn.append(nameField)

      // Create board running different version text
      let wrong = 0;
      let offline = 0;
      let boardCount = 0;
      for (const board of Object.values(value.boards)) {
        if (board.online && board.version !== value.version) {
          wrong++
        }
        if (!board.online) {
          offline++
        }
        boardCount++
      }

      let differentVersionLine = ""
      if (wrong > 0) {
        differentVersionLine = `<span style="color: black"><b>Boards running a different version:</b> ${wrong}</span>`
        if (wrong > 2) {
          container.css("background-color", "#ffc676")
        }
      }
      let offlineLine = ""
      if (offline > 0) {
        offlineLine = `<span style="color: black"><b>Boards offline:</b> ${offline}</span>`
        if (offline > 2) {
          container.css("background-color", "#ff7676")
        }
      }

      const groupBox = $(`
                <div>
                    <b>Group ID:</b> ${key} | <b>Current Version:</b> ${value.version} <br>
                    <b>Boards in group:</b> ${Object.values(value.boards).length} <br>
                    ${differentVersionLine}
                    ${offlineLine}
                </div>
            `)
      leftColumn.append(groupBox)
      const rightColumn = $('<div class="col-3"></div>')
      container.append(rightColumn)

      // Buttons
      const rename = $(`<button type="button" class="btn btn-primary" style="margin-top: 5px ;margin-bottom: 10px">Rename Group</button>`)
      const updateBoards = $(`<button type="button" class="btn btn-primary" style=" margin-top: 5px ;margin-bottom: 10px">Upload to Boards</button>`)
      const deleteButton = $(`<button type="button" class="btn btn-danger" style="margin-top: 5px ;margin-bottom: 10px">Delete group</button>`)
      const onButton = $(`<button type="button" class="btn btn-success" style="margin-top: 5px ;margin-bottom: 10px">Turn On group</button>`)
      const offButton = $(`<button type="button" class="btn btn-warning" style="margin-top: 5px ;margin-bottom: 10px">Turn Off group</button>`)
      const resetButton = $(`<button type="button" class="btn btn-primary" style="">Reset</button>`)

      // Set name handler
      rename.on('click', (e) => {
        const name = prompt("Provide group Name", "")
        if (!name || name === "")
          return;

        fetch("/api/group-name", {
          method: "POST",
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            name: name,
            group: key
          })
        }).then(response => response.json()).then(data => {
            if (data.message === "DONE!") {
              nameField.text(name)
              groups[key].name = name
              updateGroups(groups)
            }
          }
        )

        e.preventDefault()
      })

      updateBoards.on('click', (e) => {
        fetch('/api/set-group-version', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            group: key,
            version: value.version,
            autoUpdate: true
          })
        }).then(res => res.json())
          .then(data => {
            if (data.message) {
              // Set the groups that are being updated inactive
              for (const groupID of data.groupIds) {
                groupUpdating(groupID);
              }
            }
          })

        e.preventDefault()
      })

      deleteButton.on('click', (e) => {
        let c = confirm(`Are you sure you want to delete group: ${name}`)
        if (c) {
          fetch('/api/delete-group', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              group: key
            })
          }).then(res => res.json())
            .then(data => {
              if (data.message) {
                // Set the groups that are being updated inactive
                deleteGroup(data.groupId);
              }
            })
        }
        e.preventDefault()
      })

      onButton.on('click', (e) => {
        fetch('/api/on-status', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            group: key,
            state: "T"
          })
        }).then(res => res.json())
            .then(data => {
              if (data.message) {
                // Set the groups that are being updated inactive
                //TODO visualize on
              }
            })

        e.preventDefault()
      })

      offButton.on('click', (e) => {
        fetch('/api/on-status', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            group: key,
            state: "F"
          })
        }).then(res => res.json())
            .then(data => {
              if (data.message) {
                // Set the groups that are being updated inactive
                //TODO visualize off
              }
            })

        e.preventDefault()
      })

      resetButton.on('click', (e) => {
        fetch('/api/reset', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            group: key
          })
        }).then(res => res.json())
            .then(data => {
              if (data.message) {
                // Set the groups that are being updated inactive
                //TODO visualize this
              }
            })

        e.preventDefault()
      })

      rightColumn.append(rename)
      if (boardCount > 0) {
        rightColumn.append(updateBoards)
        rightColumn.append(onButton)
        rightColumn.append(offButton)
        rightColumn.append(resetButton)
      } else {
        rightColumn.append(deleteButton)
      }


      groupsDiv.append("<hr>")

    }

    // Update the Group views
    updateGroups(groups);
  }

  // Update the Re-Upload file list
  if (typeof files !== "undefined") {
    const fileSelect = $("#fileSelect")
    fileSelect.html("");
    files.sort((f1, f2) => {
      const d1 = new Date(f1.time)
      const d2 = new Date(f2.time)
      return d1 - d2
    })
    files.reverse()
    for (let file of files) {
      if (String(file.file).indexOf(".bin") !== -1) {
        file.file = String(file.file).split(".bin")[0];
        const d = new Date(file.time)
        fileSelect.append(`<option value="${file.file}"> ${d.getUTCDate()}/${d.getUTCMonth() + 1}/${d.getUTCFullYear()} - ${d.toLocaleTimeString().replace(/:\\d+ /, ' ')} - ${file.file}</option>`)
      }
    }
  }

  if (typeof alerts !== "undefined") {
    const counter = $("#alertCounter")

    alertCounter += alerts.length

    if (alertCounter > 0) {
      counter.text(alertCounter)
    } else {
      counter.text("")
    }

    const alertBox = $("#alertBox")
    for (let a of alerts) {
      console.log(a);
      let icon = '<div class="icon-circle bg-primary"><i class="fas fa-robot text-white"></i></div>'
      if (a.type === "update") {
        icon = '<div class="icon-circle bg-info"><i class="fas fa-pencil text-white"></i></div>'
      } else if (a.type === "warning") {
        icon = '<div class="icon-circle bg-warning"><i class="fas fa-warning text-white"></i></div>'
      }

      const currently = new Date(a.time);
      const element = $(`<a class="dropdown-item d-flex align-items-center" href="#"></a>`);
      element.append(`
            <div class="mr-3">
                ${icon}
            </div>
            <div>
                <div class="small text-gray-500">${currently.getHours() + ":"
      + currently.getMinutes() + ":"
      + currently.getSeconds()}</div>
                <span class="font-weight-bold">${a.message}</span>
            </div>`)

      element.on("click", () => {
        element.remove();
        alertCounter--;
        if (alertCounter > 0) {
          counter.text(alertCounter)
        } else {
          counter.text("")
        }
      });

      alertBox.prepend(element)
    }

  }
}

function deleteGroup(groupId) {
  $(`#group-${groupId}`).remove()
}

function groupUpdating(groupId) {
  const groupContainer = $(`#group-${groupId}`)
  const buttons = $(`#group-${groupId} > div > button`)

  buttons.prop('disabled', true);
  console.log("disabled!")
  groupContainer.css("background-color", "#e0e0e0")


  // Re enable after 10 seconds
  setTimeout(() => {
    buttons.prop('disabled', false);
    groupContainer.css("background-color", "transparent")
    console.log("enabled!")
  }, 10000)
}
