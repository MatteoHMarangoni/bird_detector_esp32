class Group {
    constructor(id,name, version) {
        this.id = id
        this.name = name
        this.version = version

        this.boards = {}
    }

    setVersion(version, autoUpdateBoards = false) {
        this.version = version;

        if (autoUpdateBoards)
            this.updateBoards()
    }

    updateBoards() {
        for (const board of Object.values(this.boards)) {
            if (board.online)
                board.updateVersion(this.version)
        }
    }

    hasBoard(board) {
        return typeof this.boards[board.mac] !== "undefined"
    }

    addBoard(board) {
        if (this.hasBoard(board)) {
            // Remove the old board
            delete this.boards[board.mac]
        }
        this.boards[board.mac] = board
    }

    getBoard(board) {
        return this.boards[board.mac]
    }

    remove(board) {
        if (this.hasBoard(board)) {
            // Remove the old board
            delete this.boards[board.mac]
        }
    }

    forAll(f, arg) {
        for (const board of Object.values(this.boards)) {
            if (board.online)
                f(board,arg)
        }
    }

    export() {
        return {
            id: this.id,
            name: this.name,
            version: this.version
        }
    }
}

module.exports = Group
