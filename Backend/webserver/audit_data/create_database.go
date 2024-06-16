package audit_data

import (
    "bufio"
    "crypto/sha256"
    "encoding/hex"
    "log"
    "os"
    "regexp"
    "strconv"

    "gorm.io/driver/postgres"
    "gorm.io/gorm"
    "gorm.io/gorm/clause"
)

var DB *gorm.DB

func InitDatabaseConnection() {
    dsn := "host=/var/run/postgresql/ user=postgres password=postgres dbname=tc_cadet_dataset_db port=5432 sslmode=disable TimeZone=Asia/Shanghai"
    var err error
    DB, err = gorm.Open(postgres.Open(dsn), &gorm.Config{})
    if err != nil {
        log.Fatal("failed to connect database")
    }

    // AutoMigrate will create the tables, adding missing columns and indexes
    if err := DB.AutoMigrate(&NetFlowNode{}, &SubjectNode{}, &FileNode{}, &NodeID{}, &Event{}); err != nil {
        log.Fatal("failed to migrate database schema")
    }
}

type NetFlowNode struct {
    gorm.Model
    UUID       string `gorm:"column:node_uuid"`
    Hash       string `gorm:"column:hash_id"`
    LocalAddr  string `gorm:"column:src_addr"`
    LocalPort  string `gorm:"column:src_port"`
    RemoteAddr string `gorm:"column:dst_addr"`
    RemotePort string `gorm:"column:dst_port"`
}

func (n *NetFlowNode) TableName() string {
    return "netflow_node_table"
}

type SubjectNode struct {
    gorm.Model
    UUID string `gorm:"column:node_uuid"`
    Hash string `gorm:"column:hash_id"`
    Exec string `gorm:"column:exec"`
}

func (s *SubjectNode) TableName() string {
    return "subject_node_table"
}

type FileNode struct {
    gorm.Model
    UUID string `gorm:"column:node_uuid"`
    Hash string `gorm:"column:hash_id"`
    Path string `gorm:"column:path"`
}

func (f *FileNode) TableName() string {
    return "file_node_table"
}

type NodeID struct {
    gorm.Model
    Hash   string `gorm:"column:hash_id;primaryKey"`
    Type   string `gorm:"column:node_type"`
    Msg    string `gorm:"column:msg"`
    Index  int64  `gorm:"column:index_id"`
}

func (n *NodeID) TableName() string {
    return "node2id"
}

func (n *NodeID) BeforeCreate(tx *gorm.DB) (err error) {
    return nil
}

type Event struct {
    gorm.Model
    SrcNode      string `gorm:"column:src_node"`
    SrcIndexID   string `gorm:"column:src_index_id"`
    Operation    string `gorm:"column:operation"`
    DstNode      string `gorm:"column:dst_node"`
    DstIndexID   string `gorm:"column:dst_index_id"`
    TimestampRec int64  `gorm:"column:timestamp_rec"`
}

func (e *Event) TableName() string {
    return "event_table"
}

func stringToSha256(originStr string) string {
    hash := sha256.New()
    hash.Write([]byte(originStr))
    return hex.EncodeToString(hash.Sum(nil))
}

type SocketOperation struct {
    TimestampRec     int64  `json:"timestamp_rec"`
    Process          string `json:"process"`
    Event            string `json:"event"`
    SourceIP         string `json:"source_ip"`
    SourcePort       int    `json:"source_port"`
    DestinationIP    string `json:"destination_ip"`
    DestinationPort  int    `json:"destination_port"`
}

func generateUUID() string {
    // 生成一个随机的 UUID
    u := uuid.New()
    
    // 将 UUID 格式化为字符串
    return u.String()
}

// storeNetFlow 函数现在接受 SocketOperation 类型的参数
func storeNetFlow(socketOps []SocketOperation, db *gorm.DB) {
    for _, socketOp := range socketOps {
        localAddr := socketOp.SourceIP
        localPort := strconv.Itoa(socketOp.SourcePort)
        remoteAddr := socketOp.DestinationIP
        remotePort := strconv.Itoa(socketOp.DestinationPort)

        nodeProperty := localAddr + "," + localPort + "," + remoteAddr + "," + remotePort
        hash := stringToSha256(nodeProperty)

        netFlowNode := NetFlowNode{
            UUID:       generateUUID(),
            Hash:       hash,
            LocalAddr:  localAddr,
            LocalPort:  socketOp.SourcePort, // Assuming SourcePort is already int
            RemoteAddr: remoteAddr,
            RemotePort: socketOp.DestinationPort, // Assuming DestinationPort is already int
        }

        // 查询是否已经存在相同hash的节点
        var count int64
        db.Model(&NetFlowNode{}).Where("hash = ?", hash).Count(&count)
        if count == 0 {
            db.Clauses(clause.OnConflict{DoNothing: true}).Create(&netFlowNode)
        }
    }
}

type FolderWatch struct {
    File         string `json:"file"`
    TimestampRec int64  `json:"timestamp_rec"`
    Process      string `json:"process"`
    Event        string `json:"event"`
}

func storeSubject(folderWatch []FolderWatch, db *gorm.DB) {
    for _, watch := range folderWatch {
        uuid := generateUUID()
        exec := watch.Process

        subjectNode := SubjectNode{
            UUID: uuid,
            Hash: stringToSha256(exec),
            Exec: exec,
        }

        // 查询是否有Exec相同的节点
        var count int64
        db.Model(&SubjectNode{}).Where("exec = ?", exec).Count(&count)
        if count == 0 {
            db.Clauses(clause.OnConflict{DoNothing: true}).Create(&subjectNode)
        }
    }
}

func storeFile(folderWatch []FolderWatch, db *gorm.DB) {
    for _, watch := range folderWatch {
        uuid := generateUUID()
        path := watch.File

        fileNode := FileNode{
            UUID: uuid,
            Hash: stringToSha256(path),
            Path: path,
        }

        // 查询是否有Path相同的节点
        var count int64
        db.Model(&FileNode{}).Where("path = ?", path).Count(&count)
        if count == 0 {
            db.Clauses(clause.OnConflict{DoNothing: true}).Create(&fileNode)
        }
    }   
}
   
func createNodeList(db *gorm.DB) (map[string]interface{}, map[string]string, map[string]string, map[string]string) {
    nodeList := make(map[string][]string)
    fileUUID2Hash := make(map[string]string)
    subjectUUID2Hash := make(map[string]string)
    netUUID2Hash := make(map[string]string)

    // File
    var fileNodes []FileNode
    db.Find(&fileNodes)
    for _, node := range fileNodes {
        nodeList[node.Hash] = []string{"file", node.Path}
        fileUUID2Hash[node.UUID] = node.Hash
    }

    // Subject
    var subjectNodes []SubjectNode
    db.Find(&subjectNodes)
    for _, node := range subjectNodes {
        nodeList[node.Hash] = []string{"subject", node.Exec}
        subjectUUID2Hash[node.UUID] = node.Hash
    }

    // NetFlow
    var netFlowNodes []NetFlowNode
    db.Find(&netFlowNodes)
    for _, node := range netFlowNodes {
        nodeList[node.Hash] = []string{"netflow", node.LocalAddr + ":" + node.LocalPort + " -> " + node.RemoteAddr + ":" + node.RemotePort}
        netUUID2Hash[node.UUID] = node.Hash
    }

    // Store node list
    var nodeListDatabase []NodeID
    nodeIndex := 0
    for hash, details := range nodeList {
        nodeListDatabase = append(nodeListDatabase, NodeID{
            Hash:  hash,
            Type:  details[0],
            Value: details[1],
            Index: nodeIndex,
        })
        nodeIndex++
    }

    db.Clauses(clause.OnConflict{DoNothing: true}).Create(&nodeListDatabase)

    var rows []NodeID
    db.Order("index").Find(&rows)
    nodeID2Msg := make(map[string]interface{})
    for _, row := range rows {
        nodeID2Msg[row.Hash] = map[string]string{row.Type: row.Value}
        nodeID2Msg[strconv.Itoa(row.Index)] = row.Hash
    }

    return nodeID2Msg, subjectUUID2Hash, fileUUID2Hash, netUUID2Hash
}

type Events struct {
    folderWatch []FolderWatch,
    socketOps   []SocketOperation,
}

func storeEvent( db *gorm.DB, events Events, processed map[string]bool, nodeID2Msg map[string]interface{}, subjectUUID2Hash map[string]string, fileUUID2Hash map[string]string, netUUID2Hash map[string]string) {
    for _, watch := range events.folderWatch {
        srcNode := subjectUUID2Hash[watch.Process]
        dstNode := fileUUID2Hash[watch.File]
        operation := watch.Event
        timestampRec := watch.TimestampRec

        if _, ok := processed[srcNode+dstNode+operation]; !ok {
            processed[srcNode+dstNode+operation] = true
            db.Create(&Event{
                SrcNode:      srcNode,
                SrcIndexID:   nodeID2Msg[srcNode].(string),
                Operation:    operation,
                DstNode:      dstNode,
                DstIndexID:   nodeID2Msg[dstNode].(string),
                TimestampRec: timestampRec,
            })
        }
    }

    for _, socketOp := range events.socketOps {
        srcNode := netUUID2Hash[socketOp.SourceIP+","+strconv.Itoa(socketOp.SourcePort)+","+socketOp.DestinationIP+","+strconv.Itoa(socketOp.DestinationPort)]
        dstNode := subjectUUID2Hash[socketOp.Process]
        operation := socketOp.Event
        timestampRec := socketOp.TimestampRec

        if _, ok := processed[srcNode+dstNode+operation]; !ok {
            processed[srcNode+dstNode+operation] = true
            db.Create(&Event{
                SrcNode:      srcNode,
                SrcIndexID:   nodeID2Msg[srcNode].(string),
                Operation:    operation,
                DstNode:      dstNode,
                DstIndexID:   nodeID2Msg[dstNode].(string),
                TimestampRec: timestampRec,
            })
        }
    }
}

func InsertEvents(events Events) {
    storeNetFlow(events.socketOps, DB)
    storeSubject(events.folderWatch, DB)
    storeFile(events.folderWatch, DB)

    nodeID2Msg, subjectUUID2Hash, fileUUID2Hash, netUUID2Hash := createNodeList(DB)

    processed := make(map[string]bool)
    storeEvent(DB, events, processed, nodeID2Msg, subjectUUID2Hash, fileUUID2Hash, netUUID2Hash)
}