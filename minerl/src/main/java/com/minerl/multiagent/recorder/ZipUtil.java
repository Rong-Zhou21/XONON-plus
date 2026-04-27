package com.minerl.multiagent.recorder;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

public class ZipUtil {

    public static void zip(String srcPath, String zipFile) {
        try {
            // todo currently this zips paths relative to current path. There must be a way to do this more generally.
            // todo can the line below be rewritten as a single statement?
            try (FileOutputStream fos = new FileOutputStream(zipFile); ZipOutputStream zos = new ZipOutputStream(fos)) {
                zipFiles(new File(srcPath), srcPath, zos);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void unzip(String zipFile, String dstPath) {
        try {
            unzipFiles(zipFile, dstPath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static List<String> listZip(String zipFile) {
        try {
            return listFiles(zipFile);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static void zipFiles(File fileToZip, String fileName, ZipOutputStream zipOut) throws IOException {
        // todo low-pri rewrite without recursion?
//        if (fileToZip.isHidden()) {
//            return;
//        }
        if (fileToZip.isDirectory()) {
            if (fileName.endsWith("/")) {
                zipOut.putNextEntry(new ZipEntry(fileName));
                zipOut.closeEntry();
            } else {
                zipOut.putNextEntry(new ZipEntry(fileName + "/"));
                zipOut.closeEntry();
            }
            File[] children = fileToZip.listFiles();
            for (File childFile : children) {
                zipFiles(childFile, fileName + "/" + childFile.getName(), zipOut);
            }
            return;
        }
//        try (FileInputStream fis = new FileInputStream(fileToZip)) {
//            ZipEntry zipEntry = new ZipEntry(fileName);
//            zipOut.putNextEntry(zipEntry);
//            byte[] bytes = new byte[1024];
//            int length;
//            while ((length = fis.read(bytes)) >= 0) {
//                zipOut.write(bytes, 0, length);
//            }
//        }

        try (FileChannel fc = FileChannel.open(fileToZip.toPath(), StandardOpenOption.READ); InputStream fis = Channels.newInputStream(fc)) {
            ZipEntry zipEntry = new ZipEntry(fileName);
            zipOut.putNextEntry(zipEntry);
            byte[] bytes = new byte[10240];
            // ByteBuffer bb = ByteBuffer.wrap(bytes);
            int length = 0;
            while ((length = fis.read(bytes)) >= 0) {
                zipOut.write(bytes, 0, length);
            }
        }
    }


    private static void unzipFiles(String zipFile, String dstDir) throws IOException {
        File destDir = new File(dstDir);
        byte[] buffer = new byte[1024];
        try (ZipInputStream zis = new ZipInputStream(new FileInputStream(zipFile))) {
            ZipEntry zipEntry = zis.getNextEntry();
            while (zipEntry != null) {
                File newFile = newFile(destDir, zipEntry);
                if (zipEntry.isDirectory()) {
                    if (!newFile.isDirectory() && !newFile.mkdirs()) {
                        throw new IOException("Failed to create directory " + newFile);
                    }
                } else {
                    // fix for Windows-created archives
                    File parent = newFile.getParentFile();
                    if (!parent.isDirectory() && !parent.mkdirs()) {
                        throw new IOException("Failed to create directory " + parent);
                    }
                    // write file content
                    FileOutputStream fos = new FileOutputStream(newFile);
                    int len;
                    while ((len = zis.read(buffer)) > 0) {
                        fos.write(buffer, 0, len);
                    }
                    fos.close();
                }
                zipEntry = zis.getNextEntry();
            }
            zis.closeEntry();
        }
    }

    private static List<String> listFiles(String zipFile) throws IOException {
        List<String> retVal = new LinkedList<>();
        try (ZipInputStream zis = new ZipInputStream(new FileInputStream(zipFile))) {
            while (true) {
                ZipEntry zipEntry = zis.getNextEntry();
                if (zipEntry == null) {
                    break;
                }
                retVal.add(zipEntry.getName().replace('\\', '/'));
            }
        }
        return retVal;
    }

    private static File newFile(File destinationDir, ZipEntry zipEntry) throws IOException {
        String zipEntryName = zipEntry.getName().replace('\\', '/');
        File destFile = new File(destinationDir, zipEntryName);

        String destDirPath = destinationDir.getCanonicalPath();
        String destFilePath = destFile.getCanonicalPath();

        if (!destFilePath.startsWith(destDirPath + File.separator)) {
            throw new IOException("Entry is outside of the target dir: " + zipEntryName);
        }
        return destFile;
    }
}
