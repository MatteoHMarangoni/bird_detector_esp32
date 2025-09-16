class Alert {
    static UPDATE = "update"
    static JOINED = "joined"
    static WARNING = "warning"
    static GROUPNAME = "groupName"
    static OFFLINE = "offline"
    static ONLINE = "online"

    constructor(type, message) {
        this.message = message
        this.type = type
        this.time = Date.now();
    }
}

module.exports = Alert
