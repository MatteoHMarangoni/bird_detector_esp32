let dataTable;

// Call the dataTables jQuery plugin
$(document).ready(function () {
    dataTable = $('#dataTable').DataTable({
      
        "columns": [
            {
                "data": null,
                "render": function (data) {
                    if (data.online)
                        return `<span style='margin-top: 4px' class='dot client-active'>Online</span>`
                    return `<span style='margin-top: 4px' class='dot client-disabled'>Offline</span>`
                },
                "width": "5%",
                "className": "dt-center"
            },
            {"data": null, "render": function (data) {
                const entry = $(`<span>${data.name}</span>`)
                if (data.online) {
                    entry.addClass("clickable")
                    entry.addClass("rename")

                }
                return entry.prop("outerHTML")
                } },
            {"data": null, "render": function (data){
                    const entry = $(`<span>${data.group}</span>`)
                    if (data.online) {
                        entry.addClass("clickable")
                        entry.addClass("regroup")
                    }
                    return entry.prop("outerHTML")
                } },
            {"data": "version"},
            {"data": "address"},
            {"data": "mac"},
			{"data": "ssid"},
            {"data": "dbm"}
        ],
        "pageLength":100
    });

    const webSocket = new WebSocket("ws://localhost:8080")

    webSocket.onmessage = (message) => {

        const temp = String(message.data)
        // We got boards
        if (temp.substr(0, 2) === "B|") {
            const data = JSON.parse(temp.substr(2))
            const clients = data.Boards
            const groups = data.Groups
            updateViews(clients, undefined)
        }

        //We got Alerts
        if (temp.substr(0, 2) === "A|") {
            const alerts = JSON.parse(temp.substr(2))
            updateViews(undefined, alerts)
        }
    }
});

/**
 * Updates the client list view.
 * @param clients - Object of clients that is present.
 * @param alerts - The alerts generated on the server.
 */
function updateViews(clients, alerts) {

    // Update the Client views
    if (typeof clients !== "undefined") {
        dataTable.clear();
        for (let [key, value] of Object.entries(clients)) {
            const row = dataTable.row.add(value).node()

            const rowElement = $(row)
            rowElement.on("click", ".rename", () => {
                rename(key)
            })
            rowElement.on("click", ".regroup", () => {
                reGroup(key)
            })
        }
        dataTable.draw()

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

function rename(key) {
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
}

function reGroup(key){
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
}