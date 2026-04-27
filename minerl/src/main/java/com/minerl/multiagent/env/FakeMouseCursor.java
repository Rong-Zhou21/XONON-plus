package com.minerl.multiagent.env;

import com.mojang.blaze3d.matrix.MatrixStack;
import com.mojang.blaze3d.platform.GlStateManager;
import com.mojang.blaze3d.systems.RenderSystem;
import net.minecraft.client.GameSettings;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.screen.Screen;
import net.minecraft.client.renderer.texture.SimpleTexture;
import net.minecraft.client.renderer.texture.TextureManager;
import net.minecraft.resources.*;
import net.minecraft.util.ResourceLocation;

import java.io.File;

public class FakeMouseCursor {
    private static FakeMouseCursor instance = new FakeMouseCursor();
    private FakeMouseCursor() {
        System.out.println("*** creating fake mouse cursor ***");
    }

    public static FakeMouseCursor getInstance() {
        return instance;
    }

    public void render(MatrixStack matrixStack, Screen screen, int x, int y) {
        GameSettings gameSettings = Minecraft.getInstance().gameSettings;
        int size= gameSettings.fakeCursorSize;
        if (size == 0) {
            return;
        }
        GlStateManager.enableTexture();
        GlStateManager.disableLighting();
        GlStateManager.disableDepthTest();
        if (screen == null) {
            return;
        }
        GlStateManager.pushMatrix();
        bindTexture(size);
        GlStateManager.enableRescaleNormal();
        GlStateManager.enableAlphaTest();

        GlStateManager.alphaFunc(516, 0.1F);
        GlStateManager.enableBlend();
        GlStateManager.blendFunc(GlStateManager.SourceFactor.SRC_ALPHA.param, GlStateManager.DestFactor.ONE_MINUS_SRC_ALPHA.param);
        RenderSystem.color4f(1.0F, 1.0F, 1.0F, 1.0F);
        screen.blit(matrixStack, x, y, 0, 0, size, size);

        GlStateManager.disableAlphaTest();
        GlStateManager.disableRescaleNormal();
        GlStateManager.disableLighting();
        GlStateManager.popMatrix();

    }


    private static void bindTexture (int size) {
        if (size < 1 || size > 16) {
            throw new RuntimeException("Cursor size should be between 1 and 16 (requested " + size + ")");
        }
        int rounded_size = 4;
        // We select nearest power of 2 larger than the size (because the cursor sprites are available for sizes that
        // are powers of two). The sprite is then cropped around top left corner is size != power of 2.
        // Also, the smallest available sprite is 4x4, so if requested size is 1 or 2, we are still cropping 4x4 sprite.
        while (rounded_size < size) {
            rounded_size <<= 1;
        }
        String textureName = new StringBuilder()
                    .append("cursors:mouse_cursor_white_")
                    .append(rounded_size)
                    .append("x")
                    .append(rounded_size)
                    .append(".png").toString();

        TextureManager tm = Minecraft.getInstance().getTextureManager();
        IResourceManager rm = Minecraft.getInstance().getResourceManager();
        ResourceLocation texLocation = new ResourceLocation(textureName);
        if (tm.getTexture(texLocation) == null) {
            IResourcePack resourcePack = new JavaResourcePack();
            ((SimpleReloadableResourceManager)rm).addResourcePack(resourcePack);
            SimpleTexture texture = new SimpleTexture(texLocation);
            tm.loadTexture(texLocation, texture);
        }
        tm.bindTexture(texLocation);
        tm.getTexture(texLocation).setBlurMipmapDirect(false, false);
    }
}
