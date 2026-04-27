package com.minerl.multiagent.recorder;

import com.microsoft.azure.storage.CloudStorageAccount;
import com.microsoft.azure.storage.blob.CloudBlobClient;
import com.microsoft.azure.storage.blob.CloudBlobContainer;
import com.microsoft.azure.storage.blob.CloudBlockBlob;
import net.minecraft.util.LoggingPrintStream;

import java.io.File;
import java.rmi.server.ExportException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.*;
import java.util.stream.Collectors;

public class AzureUpload {
    private static final int maxRetries = 3;
    private static ExecutorService executor = Executors.newFixedThreadPool(10);

    public static boolean upload(String localPath, String azurePath) {
        String azureSas = System.getenv("AZURE_SAS");
        if (azureSas == null) {
            System.out.println("No shared access signature (SAS) found, skipping upload of " + localPath);
            return false;
        }
        String azureFullPath = azurePath + new File(localPath).getName();
        CloudBlockBlob bcc = getBlobContainerClient(azureSas, azureFullPath);

        int attempt = 0;
        while (true) {
            try {
                System.out.println("uploading " + localPath);
                bcc.uploadFromFile(localPath);
                System.out.println(localPath + " -> " + azureFullPath + " upload successful!");
                return true;
            } catch (Exception e) {
                if (attempt == maxRetries) {
                    throw new RuntimeException(e);
                }
                System.out.println(localPath + " -> " + azureFullPath + " upload failed with " + e.toString() + ", sleeping and retrying (attempt " + attempt + " out of " + maxRetries + ")");
                try {
                    Thread.sleep(10000);
                } catch (InterruptedException e1) {
                    System.out.println("Upload thread sleep interrupted!");
                }
                attempt++;
            }
        }
    }

    public static Future<String> uploadAsync(String localPath, String azurePath) {
        return executor.submit(() -> {
            upload(localPath, azurePath);
            return azurePath;
        });
    }

    public static void finish() {
        try {
            executor.shutdown();
            executor.awaitTermination(10, TimeUnit.MINUTES);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    private static CloudBlockBlob getBlobContainerClient(String sas, String path) {
        // Parse the connection string and create a blob client to interact with Blob storage
        String connectStr = buildConnectionString(sas, path);
        try {
            CloudStorageAccount storageAccount = CloudStorageAccount.parse(connectStr);
            CloudBlobClient blobClient = storageAccount.createCloudBlobClient();
            CloudBlobContainer container = blobClient.getContainerReference(getContainerFromPath(path));
            return container.getBlockBlobReference(getBlobFromPath(path));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        /*
        BlobServiceClient blobServiceClient = new BlobServiceClientBuilder().connectionString(connectStr).buildClient();

        return blobServiceClient.getBlobContainerClient(getContainerFromPath(path));

         */
    }

    private static String buildConnectionString(String sas, String azurePath) {
        String storageAcc = getStorageAccFromPath(azurePath);
        return String.format("BlobEndpoint=https://%s.blob.core.windows.net;SharedAccessSignature=%s", storageAcc, sas);
    }

    private static String getBlobFromPath(String azurePath) {
        testAzurePath(azurePath);
        return Arrays.stream(azurePath.split("/")).skip(4).collect(Collectors.joining("/"));
    }

    private static void testAzurePath(String azurePath) {
        if (!azurePath.startsWith("az://")) {
            throw new RuntimeException("Azure path must start with az://");
        }
    }

    private static String getContainerFromPath(String azurePath) {
        testAzurePath(azurePath);
        return Arrays.stream(azurePath.split("/")).skip(3).limit(1).collect(Collectors.joining("/"));
    }

    private static String getStorageAccFromPath(String azurePath) {
        testAzurePath(azurePath);
        return Arrays.stream(azurePath.split("/")).skip(2).limit(1).collect(Collectors.joining("/"));

    }

    public static ExecutorService getExecutor() {
        return executor;
    }
}
