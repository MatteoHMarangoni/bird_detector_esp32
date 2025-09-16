const express = require('express');
const router = express.Router();
const {boards, groups, setName, setGroup, setGroupName, updateFileList, updateClients, saveGroups, deleteGroup, sendStatus, sendReset} = require('../OTA/server')
const multer = require("multer")
const fs = require("fs")

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, './files')
  },
  filename: function (req, file, cb) {
    const name = getName("./files/",file.originalname)
    req.newFileName = name;
    cb(null, name + ".bin")
  }
})

const upload = multer({storage: storage})

function getName(folder, filename, increment = 0){
  if (increment) {
    if (fs.existsSync(`${folder}${filename}-${increment?increment:""}.bin`)) {
      return getName(folder, filename, ++increment)
    }
    return `${filename}-${increment?increment:""}`
  }
  if (fs.existsSync(`${folder}${filename}.bin`)) {
    return getName(folder, filename, ++increment)
  }
  return `${filename}`

}


/**
 * Returns a list of all the connected clients.
 */
router.get("/clients", ((req, res) => {
  res.json(boards)
}))

router.post("/name", ((req, res) => {
  const client = req.body.client;
  const name = req.body.name;

  setName(boards[client].address, name);

  res.send("Name set")
}))

router.post("/group-name", ((req, res) => {
  const group = req.body.group;
  const name = req.body.name;

  setGroupName(group, name);

  res.json({message: "DONE!"})
}))

router.post("/group", ((req, res) => {
  const client = req.body.client;
  const group = req.body.group;

  setGroup(boards[client].address, group);

  res.send("group set")
}))

router.post("/on-status", (req, res) => {
  const groupId = req.body.group;
  const state = req.body.state;

  // Set new versions for all the groups

    if (typeof groups[groupId] !== "undefined") {
      groups[groupId].forAll(sendStatus, state)
    }

  res.json({message: `Group's on-state is now set to ${state}`})
});

router.post("/reset", (req, res) => {
  const groupId = req.body.group;

  // Set new versions for all the groups
  if (typeof groups[groupId] !== "undefined") {
    groups[groupId].forAll(sendReset)
  }

  res.json({message: `Group's on-state is now set to ${state}`})
});

/**
 * Uses and already uploaded file and sends the go to all the selected clients.
 */
router.post("/set-group-version", (req, res) => {
  const groupIds = req.body.group;
  const version = req.body.version;
  const update = req.body.autoUpdate;

  const updated = [];

  // Set new versions for all the groups
  for (const groupId of groupIds) {
    if (typeof groups[groupId] !== "undefined") {
      groups[groupId].setVersion(version,update)
      updated.push(groupId)
    }
  }

  updateClients()
  saveGroups()
  res.json({message: "Group(s) successfully updated!", groupIds: updated})
});

/**
 * Uses and already uploaded file and sends the go to all the selected clients.
 */
router.post("/delete-group", (req, res) => {
  const groupId = req.body.group;

  deleteGroup(groupId)

  updateClients()
  saveGroups()
  res.json({message: "Group deleted", groupId: groupId})
});

/**
 * Upload a file and sends the go to all the selected clients.
 */
router.post("/upload", upload.single("file"), (req, res) => {
  const fileName = req.newFileName;

  res.json({fileName: fileName});
  updateFileList();
});

/**
 * Get the file.
 * Called by the client when the need the file.
 */
router.get("/file/:version", function (req, res) {
  const fileName = (req.params.version) ? req.params.version : "test.bin"
  const file = `${__dirname}/../files/${fileName}.bin`;
  res.download(file);
});

module.exports = router;
